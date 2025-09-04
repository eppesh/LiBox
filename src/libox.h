#include <immintrin.h>
#include <omp.h>
#include <xmmintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "segmentation.h"

#define SORT_BOX

// for Box lock, define one
#define LOCK_SEARCH
//#define LOCK_SEARCH_SPIN_LOCK

// for Segment lock, define one
#define LOCK_SEG
//#define LOCK_SEG_LOAD_ONCE

// for EBR lock, define one
#define LOCK_EBR

#define overflowCapacity 3
#define maxKey 64

#define NUM_BOXES_TO_LOOK 3

volatile int dummy;
using namespace std;

namespace liboxns {

void exponential_backoff(int retry_count);


class ThreadIdManager {
private:
    static thread_local int cached_thread_id_;
    static int max_thread_num_;

public:
    static void initialize(int max_threads) {
        max_thread_num_ = max_threads;
    }

    static int get_thread_id() {
        if (cached_thread_id_ == -1) {
            cached_thread_id_ = omp_get_thread_num();
            if (cached_thread_id_ >= max_thread_num_) {
                cached_thread_id_ = cached_thread_id_ % max_thread_num_;
            }
        }
        return cached_thread_id_;
    }

    static void refresh_cache() {
        cached_thread_id_ = -1;
    }
};

thread_local int ThreadIdManager::cached_thread_id_ = -1;
int ThreadIdManager::max_thread_num_ = 0;

class ThreadLocalCounter {
private:
    struct alignas(64) ThreadCounter {
        std::atomic<uint32_t> count{0};
        char padding[60];
    };

    std::unique_ptr<ThreadCounter[]> thread_counters_;
    int thread_num_;
public:
    explicit ThreadLocalCounter(int thread_num)
        : thread_num_(thread_num) {
        thread_counters_ = std::make_unique<ThreadCounter[]>(thread_num);
    }

    void increment() {
        int slot = ThreadIdManager::get_thread_id();
        thread_counters_[slot].count.fetch_add(1, std::memory_order_relaxed);
    }

    void decrement() {
        int slot = ThreadIdManager::get_thread_id();
        thread_counters_[slot].count.fetch_sub(1, std::memory_order_relaxed);
    }

    bool is_zero() const {
        for (int i = 0; i < thread_num_; i++) {
            if (thread_counters_[i].count.load(std::memory_order_acquire) > 0) {
                return false;
            }
        }
        return true;
    }

    int64_t get_total_count() const {
        int64_t total = 0;
        for (int i = 0; i < thread_num_; i++) {
            total += thread_counters_[i].count.load(std::memory_order_acquire);
        }
        return total;
    }
};

// Thread-local timing utility for measuring wait operations
class ThreadLocalWaitTimingStats {
private:
    struct alignas(64) ThreadTimingStats {
        uint64_t total_wait_count{0};
        uint64_t total_wait_time_ns{0};
        uint64_t max_wait_time_ns{0};
        char padding[40]; // Ensure 64-byte alignment
    };

    std::unique_ptr<ThreadTimingStats[]> thread_stats_;
    int thread_num_;
    std::string name_;

public:
    explicit ThreadLocalWaitTimingStats(int thread_num, const std::string& name = "")
        : thread_num_(thread_num), name_(name) {
        thread_stats_ = std::make_unique<ThreadTimingStats[]>(thread_num);
    }

    void record_wait(uint64_t wait_time_ns) {
        int slot = ThreadIdManager::get_thread_id();
        assert(slot < thread_num_);
        auto& stats = thread_stats_[slot];
        stats.total_wait_count++;
        stats.total_wait_time_ns += wait_time_ns;
        if (wait_time_ns > stats.max_wait_time_ns) {
            stats.max_wait_time_ns = wait_time_ns;
        }
    }

    void print_stats() {
        uint64_t total_count = 0;
        uint64_t total_time = 0;
        uint64_t max_time = 0;
        int active_threads = 0;

        for (int i = 0; i < thread_num_; i++) {
            const auto& stats = thread_stats_[i];
            if (stats.total_wait_count > 0) {
                active_threads++;
            }
            total_count += stats.total_wait_count;
            total_time += stats.total_wait_time_ns;
            if (stats.max_wait_time_ns > max_time) {
                max_time = stats.max_wait_time_ns;
            }
        }

        if (total_count > 0) {
            double avg_time_us = static_cast<double>(total_time) / total_count / 1000.0;
            double max_time_us = static_cast<double>(max_time) / 1000.0;
            double total_time_us = static_cast<double>(total_time) / 1000.0;

            std::cout << "[" << name_ << "] Wait Stats: "
                      << "count=" << total_count << ", "
                      << "active_threads=" << active_threads << "/" << thread_num_ << ", "
                      << "avg_time=" << std::fixed << std::setprecision(2) << avg_time_us << "us, "
                      << "max_time=" << max_time_us << "us, "
                      << "total_time=" << total_time_us << "us" << std::endl;
        } else {
            std::cout << "[" << name_ << "] Wait Stats: count=0 (no waits recorded)" << std::endl;
        }
    }
};

ThreadLocalWaitTimingStats exponential_backoff_stats(84, "Exponential backoff");

inline void exponential_backoff(int retry_count) {
    auto backoff_start = std::chrono::high_resolution_clock::now();
    if (retry_count > 10) retry_count = 10;
    int backoff = (1 << retry_count);
    std::this_thread::sleep_for(std::chrono::microseconds(backoff));
    //std::this_thread::yield();
    auto backoff_end = std::chrono::high_resolution_clock::now();
    auto backoff_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(backoff_end - backoff_start).count();
    exponential_backoff_stats.record_wait(backoff_duration);
}

enum class InsertStatus {
    SUCCESS,
    FULL,
    SPLIT,
    OUT_OF_RANGE,
    ERROR
};

struct InsertResult {
    InsertStatus status;
    int box_index;
};

enum class DeleteStatus {
    SUCCESS,
    NOT_FOUND,
    SPLIT,
    OUT_OF_RANGE,
    ERROR
};

struct DeleteResult {
    DeleteStatus status;
    bool found;
};

enum class SearchStatus {
    SUCCESS,
    NOT_FOUND,
    SPLIT,
    OUT_OF_RANGE,
    ERROR
};

template <typename KeyType, typename ValueType>
struct SearchResult {
    SearchStatus status;
    ValueType value;
};

static constexpr int32_t BELOW_LOWER_BOUND = -1;
static constexpr int32_t ABOVE_UPPER_BOUND = -2;

template <typename KeyType, typename ValueType>
class OverflowKeyValue {
   private:
    size_t maxSize = 0;
    size_t validSize = 0;
    size_t nearestEmptySlot  = 0;
    bitset<maxKey> valid_flags;

    alignas(64) array<ValueType, maxKey> values;
    alignas(64) array<KeyType, maxKey> keys;
    alignas(64) array<uint8_t, maxKey> keys_low;

    size_t avx512_filter_keys_optimized(KeyType key_low_bound,
                                        pair<KeyType, ValueType>* result_buffer,
                                        size_t max_results) const {
        size_t collected = 0;
        uint8_t target_low = static_cast<uint8_t>(key_low_bound & 0xFF);
        __m512i v_threshold = _mm512_set1_epi8(target_low);

        __m512i v_keys = _mm512_load_si512(reinterpret_cast<const __m512i*>(keys_low.data()));
        __mmask64 mask = _mm512_cmpge_epi8_mask(v_keys, v_threshold);

        while (mask && collected < max_results) {
            int pos = __builtin_ctzll(mask);
            result_buffer[collected++] = {keys[pos], values[pos]};
            mask &= mask - 1;
        }
        return collected;
    }

    size_t copyDataWithLimit(pair<KeyType, ValueType>* result, size_t max_count) const {
        size_t to_copy = min(maxSize, max_count);
        for (size_t i = 0; i < to_copy; i++) {
            result[i] = {keys[i], values[i]};
        }
        return to_copy;
    }

    void updateNearestEmptySlot() {
        for (size_t i = nearestEmptySlot; i < maxKey; i++) {
            if (!valid_flags[i]) {
                nearestEmptySlot = i;
                return;
            }
        }
        nearestEmptySlot = maxKey;
    }
   public:
    OverflowKeyValue() {}

    size_t getTotalCount() const {
        return maxSize;
    }

    bool hasEmptySlots() const {
        return nearestEmptySlot < maxKey;
    }

    size_t findKeyIndex(KeyType key) const {
        uint8_t key_low = key & 0xFF;
        uint8_t target_low = ((key_low * 251) % 255) + 1;

        __m512i v_target_low = _mm512_set1_epi8(target_low);
        __m512i v_keys_low = _mm512_load_si512(reinterpret_cast<const __m512i*>(keys_low.data()));
        __mmask64 mask_low = _mm512_cmpeq_epi8_mask(v_keys_low, v_target_low);

        while (mask_low) {
            size_t candidate = __builtin_ctzll(mask_low);
            mask_low &= mask_low - 1;
            if (keys[candidate] == key && valid_flags[candidate]) return (candidate < maxSize) ? candidate : -1;
        }
        return maxKey;
    }

    void updateValueAt(int index, ValueType value) {
        values[index] = value;
    }

    bool deleteKey(KeyType key) {
        size_t index = findKeyIndex(key);
        if (index != maxKey) {
            valid_flags[index] = 0;
            validSize--;
            nearestEmptySlot = index < nearestEmptySlot ? index : nearestEmptySlot;
            return true;
        }
        return false;
    }

    InsertResult insert(KeyType key, ValueType value) {
        if (nearestEmptySlot >= maxKey) {
            return {InsertStatus::FULL, -1};
        }
        keys[nearestEmptySlot] = key;
        values[nearestEmptySlot] = value;
        uint8_t key_low = key & 0xFF;
        keys_low[nearestEmptySlot] = ((key_low * 251) % 255) + 1;
        valid_flags[nearestEmptySlot] = true;
        validSize++;
        if (nearestEmptySlot == maxSize) {
            maxSize++;
        }
        updateNearestEmptySlot();
        return {InsertStatus::SUCCESS, -1};
    }

    SearchResult<KeyType, ValueType> search(KeyType key) {
        size_t index = findKeyIndex(key);
        if (index != maxKey) {
            return {SearchStatus::SUCCESS, values[index]};
        }
        return {SearchStatus::NOT_FOUND, -1};
    }

    size_t size() const {
        return maxSize;
    }

    vector<pair<KeyType, ValueType>> getEntries() const {
        vector<pair<KeyType, ValueType>> entries;
        for (size_t i = 0; i < maxSize; i++) {
            entries.push_back({keys[i], values[i]});
        }
        return entries;
    }

    void getEntriesInPlace(vector<pair<KeyType, ValueType>>* entries) const {
        size_t start_size = entries->size();
        entries->resize(start_size + maxSize);
        for (size_t i = 0; i < maxSize; i++) {
            (*entries)[start_size + i] = {keys[i], values[i]};
        }
    }

    void getEntriesInPlace(vector<pair<KeyType, ValueType>>* entries, size_t start_pos) const {
        for (size_t i = 0; i < maxSize; i++) {
            (*entries)[start_pos + i] = {keys[i], values[i]};
        }
    }

    std::unique_ptr<OverflowKeyValue<KeyType, ValueType>> clone() const {
        auto newObj = std::make_unique<OverflowKeyValue<KeyType, ValueType>>();
        newObj->maxSize = maxSize;
        newObj->validSize = validSize;
        newObj->nearestEmptySlot = nearestEmptySlot;
        newObj->valid_flags = valid_flags;

        for (size_t i = 0; i < maxSize; i++) {
            newObj->keys[i] = keys[i];
            newObj->keys_low[i] = keys_low[i];
            newObj->values[i] = values[i];
        }
        return newObj;
    }

    size_t copyAllData(pair<KeyType, ValueType>* result) const {
        size_t i = 0;
        for (; i + 8 <= maxSize; i += 8) {
            for (int j = 0; j < 8; j++) {
                result[i + j] = {keys[i + j], values[i + j]};
            }
        }
        for (; i < maxSize; i++) {
            result[i] = {keys[i], values[i]};
        }
        return maxSize;
    }
};

struct tas_lock {
    std::atomic<bool> lock_ = {false};

    void lock() {
        for (;;) {
            if (!lock_.exchange(true, std::memory_order_acquire)) {
              break;
            }
            while (lock_.load(std::memory_order_relaxed)) {
              __builtin_ia32_pause();
            }
        }
    }

    void unlock() { lock_.store(false); }
};

template <typename KeyType, typename ValueType>
class Box {
   private:
    size_t maxSize = 0;
    size_t validSize = 0;
    size_t nearestEmptySlot = 0;
    bitset<maxKey> valid_flags;

#ifdef LOCK_SEARCH_SPIN_LOCK
    tas_lock lock_;
#endif

    mutable std::atomic<uint32_t> version_lock_{0};
    static constexpr uint32_t WRITE_LOCK_BIT = 0x80000000;
    static constexpr uint32_t VERSION_MASK   = 0x7FFFFFFF;

    alignas(64) array<KeyType, maxKey> keys;
    alignas(64) array<uint8_t, maxKey> keys_low;
    alignas(64) array<ValueType, maxKey> values;

    struct BoxSearchResult {
        uint8_t isUpdate; // 0 for update, 1 for insert, 2 for full
        size_t level;
        size_t slot;
    };

    enum BoxInsertResult : uint8_t {
        UPDATE = 0,
        INSERT = 1,
        FULL = 2
    };

    // for overflowbox
    size_t capacity;
    array<unique_ptr<OverflowKeyValue<KeyType, ValueType>>, overflowCapacity> data;

    inline bool test_write_lock(uint32_t &version) const {
        version = version_lock_.load(std::memory_order_acquire);
        return (version & WRITE_LOCK_BIT) != 0;
    }

    inline bool version_changed(uint32_t old_version) const {
        uint32_t current = version_lock_.load(std::memory_order_acquire);
        return old_version != current;
    }

    inline bool try_acquire_write_lock() {
        uint32_t expected = version_lock_.load(std::memory_order_acquire);
        if (expected & WRITE_LOCK_BIT) {
            return false;
        }
        uint32_t desired = expected | WRITE_LOCK_BIT;
        return version_lock_.compare_exchange_strong(expected, desired,
                                                    std::memory_order_acq_rel,
                                                    std::memory_order_acquire);
    }

    inline void release_write_lock() {
        uint32_t current = version_lock_.load(std::memory_order_relaxed);
        uint32_t new_version = current + 1 - WRITE_LOCK_BIT;
        version_lock_.store(new_version, std::memory_order_release);
    }

    size_t findKeyIndex(KeyType key) const {
        uint8_t key_low = key & 0xFF;
        uint8_t target_low = ((key_low * 251) % 255) + 1;

        __m512i v_target_low = _mm512_set1_epi8(target_low);
        __m512i v_keys_low = _mm512_load_si512(reinterpret_cast<const __m512i*>(keys_low.data()));
        __mmask64 mask_low = _mm512_cmpeq_epi8_mask(v_keys_low, v_target_low);

        while (mask_low) {
            size_t candidate = __builtin_ctzll(mask_low);
            mask_low &= mask_low - 1;
            if (keys[candidate] == key && valid_flags[candidate]) return (candidate < maxSize) ? candidate : -1;
        }
        return maxKey;
    }

    void updateNearestEmptySlot() {
        for (size_t i = nearestEmptySlot; i < maxKey; i++) {
            if (!valid_flags[i]) {
                nearestEmptySlot = i;
                return;
            }
        }

        if (maxSize < maxKey) {
            nearestEmptySlot = maxSize;
        } else {
            nearestEmptySlot = maxKey;
        }
    }

    size_t avx512_filter_main_keys_optimized(KeyType key_low_bound,
                                             pair<KeyType, ValueType>* result_buffer,
                                             size_t max_results) const {
        size_t collected = 0;
        uint8_t target_low = static_cast<uint8_t>(key_low_bound & 0xFF);
        __m512i v_threshold = _mm512_set1_epi8(target_low);

        __m512i v_keys = _mm512_load_si512(reinterpret_cast<const __m512i*>(keys_low.data()));
        __mmask64 mask = _mm512_cmpge_epi8_mask(v_keys, v_threshold);

        while (mask && collected < max_results) {
            int pos = __builtin_ctzll(mask);
            result_buffer[collected++] = {keys[pos], values[pos]};
            mask &= mask - 1;
        }
        return collected;
    }

    size_t copyMainKeysWithLimit(pair<KeyType, ValueType>* result, size_t max_count) const {
        size_t to_copy = std::min(maxSize, max_count);
        for (size_t i = 0; i < to_copy; i++) {
            result[i] = {keys[i], values[i]};
        }
        return to_copy;
    }

    BoxSearchResult findKeyOrSlot(KeyType key, ValueType value) {
        size_t existingIndex = findKeyIndex(key);
        if (existingIndex != maxKey) {
            values[existingIndex] = value;
            return {BoxInsertResult::UPDATE, 0, existingIndex};
        }

        if (validSize < maxKey) {
            for (size_t i = 0; i < capacity; i++) {
                size_t existingIndex = data[i]->findKeyIndex(key);
                if (existingIndex != maxKey) {
                    data[i]->updateValueAt(existingIndex, value);
                    return {BoxInsertResult::UPDATE, i+1, existingIndex};
                }
            }
            keys[nearestEmptySlot] = key;
            values[nearestEmptySlot] = value;
            uint8_t key_low = key & 0xFF;
            keys_low[nearestEmptySlot] = ((key_low * 251) % 255) + 1;
            valid_flags[nearestEmptySlot] = true;
            validSize++;
            if (nearestEmptySlot == maxSize) {
                maxSize++;
            }
            updateNearestEmptySlot();
            return {BoxInsertResult::INSERT, 0, nearestEmptySlot};
        } else {
            size_t insertBox = capacity;
            for (int i = capacity-1; i >= 0; i--) {
                size_t existingIndex = data[i]->findKeyIndex(key);
                if (existingIndex != maxKey) {
                    data[i]->updateValueAt(existingIndex, value);
                    return {BoxInsertResult::UPDATE, static_cast<size_t>(i+1), existingIndex};
                }
                if (data[i]->hasEmptySlots()) {
                    insertBox = i;
                }
            }
            if (insertBox == capacity) {
                if (capacity == overflowCapacity) {
                    return {BoxInsertResult::FULL, 0, 0}; // No space left
                }
                data[capacity] = make_unique<OverflowKeyValue<KeyType, ValueType>>();
                data[capacity]->insert(key, value);
                capacity++;
                return {BoxInsertResult::INSERT, capacity, 0};
            } else {
                data[insertBox]->insert(key, value);
                return {BoxInsertResult::INSERT, insertBox+1, 0};
            }
        }
    }

   public:
    Box() : capacity(1) {
        data[0] = make_unique<OverflowKeyValue<KeyType, ValueType>>();
    }

    Box(const Box& other)
        : maxSize(other.maxSize),
          validSize(other.validSize),
          keys_low(other.keys_low),
          valid_flags(other.valid_flags),
          nearestEmptySlot(other.nearestEmptySlot),
          capacity(other.capacity) {
        while (true) {
            uint32_t version_start;
            if (other.test_write_lock(version_start)) {
                std::this_thread::yield();
                continue;
            }
            keys = other.keys;
            keys_low = other.keys_low;
            values = other.values;
            if (!other.version_changed(version_start)) {
                break;
            }
        }
        for (size_t i = 0; i < capacity; i++) {
            if (other.data[i]) {
                data[i] = other.data[i]->clone();
            }
        }
    }

    Box& operator=(const Box& other) {
        if (this != &other) {
            while (!try_acquire_write_lock()) {
                std::this_thread::yield();
            }
            while (true) {
                uint32_t version_start;
                if (other.test_write_lock(version_start)) {
                    std::this_thread::yield();
                    continue;
                }
                maxSize = other.maxSize;
                validSize = other.validSize;
                nearestEmptySlot = other.nearestEmptySlot;
                valid_flags = other.valid_flags;
                capacity = other.capacity;
                keys = other.keys;
                keys_low = other.keys_low;
                values = other.values;
                if (!other.version_changed(version_start)) {
                    break;
                }
            }
            for (size_t i = 0; i < capacity; i++) {
                if (other.data[i]) {
                    data[i] = other.data[i]->clone();
                }
            }
            release_write_lock();
        }
        return *this;
    }

    Box(Box&& other) noexcept
        : maxSize(other.maxSize),
        validSize(other.validSize),
        nearestEmptySlot(other.nearestEmptySlot),
        valid_flags(other.valid_flags),
        capacity(other.capacity),
        keys(std::move(other.keys)),
        keys_low(std::move(other.keys_low)),
        values(std::move(other.values)) {

        version_lock_.store(other.version_lock_.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);

        for (size_t i = 0; i < capacity; i++) {
            data[i] = std::move(other.data[i]);
        }
    }

    Box& operator=(Box&& other) noexcept {
        if (this != &other) {
            while (!try_acquire_write_lock()) {
                std::this_thread::yield();
            }

            maxSize = other.maxSize;
            validSize = other.validSize;
            nearestEmptySlot = other.nearestEmptySlot;
            valid_flags = other.valid_flags;
            capacity = other.capacity;
            keys = std::move(other.keys);
            keys_low = std::move(other.keys_low);
            values = std::move(other.values);

            for (size_t i = 0; i < capacity; i++) {
                data[i] = std::move(other.data[i]);
            }

            release_write_lock();
        }
        return *this;
    }
    ~Box() = default;

    bool hasEmptySlots() const {
        while (true) {
            uint32_t version_start;
            if (test_write_lock(version_start)) {
                std::this_thread::yield();
                continue;
            }
            bool result = nearestEmptySlot < maxKey;
            if (!version_changed(version_start)) {
                return result;
            }
        }
    }

    size_t getTotalCount() const {
        size_t total = maxSize;
        for (size_t i = 0; i < capacity; i++) {
            total += data[i]->getTotalCount();
        }
        return total;
    }

    DeleteResult deleteKey(KeyType key) {
        int retry_count = 0;

    retry_delete:
        uint32_t expected = version_lock_.load(std::memory_order_acquire);
        if (expected & WRITE_LOCK_BIT) {
            exponential_backoff(retry_count++);
            goto retry_delete;
        }

        uint32_t desired = expected | WRITE_LOCK_BIT;
        if (!version_lock_.compare_exchange_strong(expected, desired,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
            exponential_backoff(retry_count++);
            goto retry_delete;
        }

        size_t index = findKeyIndex(key);
        bool found = false;
        if (index != maxKey) {
            valid_flags[index] = 0;
            validSize--;
            nearestEmptySlot = index < nearestEmptySlot ? index : nearestEmptySlot;
            found = true;
        } else {
            for (size_t i = 0; i < capacity; i++) {
                if (data[i] && data[i]->deleteKey(key)) {
                    found = true;
                    break;
                }
            }
        }

        uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
        version_lock_.store(new_version, std::memory_order_release);
        return {found ? DeleteStatus::SUCCESS : DeleteStatus::NOT_FOUND, found};
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) {
        int retry_count = 0;

    retry_write:
        uint32_t expected = version_lock_.load(std::memory_order_acquire);
        if (expected & WRITE_LOCK_BIT) {
            exponential_backoff(retry_count++);
            goto retry_write;
        }

        uint32_t desired = expected | WRITE_LOCK_BIT;
        if (!version_lock_.compare_exchange_strong(expected, desired,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
            exponential_backoff(retry_count++);
            goto retry_write;
        }

        BoxSearchResult ret = findKeyOrSlot(key, value);
        InsertStatus status = (ret.isUpdate <= 1) ? InsertStatus::SUCCESS : InsertStatus::FULL;

        uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
        version_lock_.store(new_version, std::memory_order_release);

        return {status, -1};
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        int retry_count = 0;
#ifdef LOCK_SEARCH
    retry_read:
        uint32_t start_version = version_lock_.load(std::memory_order_acquire);
        if (start_version & WRITE_LOCK_BIT) {
            exponential_backoff(retry_count++);
            goto retry_read;
        }
#endif

#ifdef LOCK_SEARCH_SPIN_LOCK
        lock_.lock();
#endif

        size_t index = findKeyIndex(key);
        ValueType result_value = -1;
        SearchStatus status = SearchStatus::NOT_FOUND;

        if (index != maxKey) {
            result_value = values[index];
            status = SearchStatus::SUCCESS;
        } else {
            status = SearchStatus::NOT_FOUND;
            for (size_t i = 0; i < capacity; i++) {
                SearchResult<KeyType, ValueType> ret = data[i]->search(key);
                if (ret.status == SearchStatus::SUCCESS) {
                    result_value = ret.value;
                    status = SearchStatus::SUCCESS;
                    break;
                }
            }
        }
#ifdef LOCK_SEARCH
        if (start_version != version_lock_.load(std::memory_order_acquire)) {
            exponential_backoff(retry_count++);
            goto retry_read;
        }
#endif
#ifdef LOCK_SEARCH_SPIN_LOCK
        lock_.unlock();
#endif
        return {status, result_value};
    }

    size_t getmaxSize() const {
        return maxSize;
    }

    vector<pair<KeyType, ValueType>> getEntries() const {
        vector<pair<KeyType, ValueType>> entries;
        for (size_t i = 0; i < maxSize; i++) {
            entries.push_back({keys[i], values[i]});
        }
        for (int i = 0; i < capacity; i++) {
            vector<pair<KeyType, ValueType>> be = data[i]->getEntries();
            entries.insert(entries.end(), be.begin(), be.end());
        }
        return entries;
    }

    void getEntriesInPlace(vector<pair<KeyType, ValueType>>* entries) const {
        size_t start_size = entries->size();
        size_t total_entries = getTotalCount();

        // Resize to accommodate all entries
        entries->resize(start_size + total_entries);

        // Assign main entries directly
        for (size_t i = 0; i < maxSize; i++) {
            (*entries)[start_size + i] = {keys[i], values[i]};
        }

        // Add overflow entries
        size_t current_pos = start_size + maxSize;
        for (int i = 0; i < capacity; i++) {
            data[i]->getEntriesInPlace(entries, current_pos);
            current_pos += data[i]->size();
        }
    }

    void getEntriesInPlace(vector<pair<KeyType, ValueType>>* entries, size_t start_pos) const {
        // Don't resize - assume the vector is already large enough
        // Assign main entries directly
        for (size_t i = 0; i < maxSize; i++) {
            (*entries)[start_pos + i] = {keys[i], values[i]};
        }

        // Add overflow entries
        size_t current_pos = start_pos + maxSize;
        for (int i = 0; i < capacity; i++) {
            data[i]->getEntriesInPlace(entries, current_pos);
            current_pos += data[i]->size();
        }
    }

    size_t getOverflowBoxCount() const {
        size_t cap;
        size_t first_overflow_size;
        cap = capacity;
        if (cap == 1) {
            first_overflow_size = data[0]->size();
            if (first_overflow_size == 0) {
                return 0;
            }
        }
        return cap;
    }
};

ThreadLocalWaitTimingStats splitting_flag_wait_stats(84, "Segment splitting_flag");
ThreadLocalWaitTimingStats split_segment_total_stats(84, "Total splitSegment time");

template <typename KeyType, typename ValueType>
class Segment {
private:
    KeyType lower_bound;
    KeyType upper_bound;
    size_t box_key_range;
    int numBoxes;
    size_t num_threads;

    mutable ThreadLocalCounter operation_counter_;
    mutable std::atomic<bool> splitting_{false};
    std::atomic_flag splitting_flag_ = ATOMIC_FLAG_INIT;
public:
    std::vector<Box<KeyType, ValueType>*> boxes;

    Segment(KeyType lower, KeyType upper, size_t box_range, int thread_num)
        : lower_bound(lower), upper_bound(upper), box_key_range(box_range),
          operation_counter_(thread_num), num_threads(thread_num) {
        size_t total = upper - lower + 1;
        size_t box_count = total / box_range;
        if (total % box_range != 0) box_count++;
        numBoxes = box_count;
        boxes.resize(box_count);
        for (size_t i = 0; i < box_count; i++) {
            boxes[i] = new Box<KeyType, ValueType>();
        }
    }

    ~Segment() {
        for (auto* box : boxes) {
            if (box != nullptr) {
                delete box;
            }
        }
    }

    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;

    Segment(Segment&& other) noexcept
        : lower_bound(other.lower_bound),
          upper_bound(other.upper_bound),
          box_key_range(other.box_key_range),
          numBoxes(other.numBoxes),
          operation_counter_(std::move(other.operation_counter_)),
          boxes(std::move(other.boxes)) {
        splitting_.store(other.splitting_.load());
    }

    Segment& operator=(Segment&& other) noexcept {
        if (this != &other) {
            lower_bound = other.lower_bound;
            upper_bound = other.upper_bound;
            box_key_range = other.box_key_range;
            numBoxes = other.numBoxes;
            operation_counter_ = std::move(other.operation_counter_);
            boxes = std::move(other.boxes);
            splitting_.store(other.splitting_.load());
        }
        return *this;
    }

    bool enter() {
        if (splitting_.load(std::memory_order_acquire)) {
            return false;
        }
        operation_counter_.increment();
        if (splitting_.load(std::memory_order_acquire)) {
            operation_counter_.decrement();
            return false;
        }
        return true;
    }

    void leave() {
        operation_counter_.decrement();
    }

    void wait_for_operations() {
        splitting_.store(true, std::memory_order_release);
        while (!operation_counter_.is_zero()) {
            //std::this_thread::yield();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) {
        if (!enter()) {
            return {InsertStatus::SPLIT, -1};
        }
        if (key < lower_bound || key >= upper_bound) {
            leave();
            return {InsertStatus::OUT_OF_RANGE, -1};
        }

        size_t box_index = (key - lower_bound) / box_key_range;
        InsertResult ret = boxes[box_index]->insertKeyValue(key, value);
        // ensure overflowing box index propagates up
        ret.box_index = static_cast<int>(box_index);
        leave();
        return ret;
    }

    DeleteResult deleteKey(KeyType key) {
        if (!enter()) {
            return {DeleteStatus::SPLIT, false};
        }
        if (key < lower_bound || key > upper_bound) {
            leave();
            return {DeleteStatus::OUT_OF_RANGE, false};
        }

        size_t box_index = (key - lower_bound) / box_key_range;
        DeleteResult result = boxes[box_index]->deleteKey(key);
        leave();
        return result;
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        //if (!enter()) {
        //    return {SearchStatus::SPLIT, -1};
        //}
        if (key < lower_bound || key > upper_bound) {
            //leave();
            return {SearchStatus::OUT_OF_RANGE, -1};
        }

        size_t box_index = (key - lower_bound) / box_key_range;
        SearchResult result = boxes[box_index]->searchKey(key);
        //leave();
        return result;
    }

    vector<pair<KeyType, ValueType>> prepare_for_split_stage1(int32_t merge_start, int32_t merge_end) {
        // Pre-calculate total size to avoid reallocations
        size_t total_size = 0;
        vector<pair<KeyType, ValueType>> mergedEntries;

        // Pre-calculate positions for each selected box
        int32_t start_box = std::max(0, merge_start);
        int32_t end_box = std::min(merge_end, static_cast<int32_t>(boxes.size()) - 1);
        if (start_box > end_box) return mergedEntries;

        std::vector<size_t> box_start_positions(static_cast<size_t>(end_box - start_box + 2), 0);
        for (int i = start_box; i <= end_box; i++) {
            size_t box_size = boxes[i]->getTotalCount();
            box_start_positions[(i - start_box) + 1] = box_start_positions[(i - start_box)] + box_size;
            total_size += box_size;
        }
        mergedEntries.resize(total_size);

        for (int i = start_box; i <= end_box; i++) {
            size_t start_pos = box_start_positions[i - start_box];
            boxes[i]->getEntriesInPlace(&mergedEntries, start_pos);
#ifdef SORT_BOX
            size_t end_pos = box_start_positions[(i - start_box) + 1];
            if (end_pos > start_pos) {
                std::sort(mergedEntries.begin() + start_pos, mergedEntries.begin() + end_pos,
                    [](const pair<KeyType, ValueType>& a, const pair<KeyType, ValueType>& b) {
                        return a.first < b.first;
                    });
            }
#endif
        }

        return mergedEntries;
    }

    vector<pair<KeyType, ValueType>> getBoxRangeEntries(int start_box, int end_box) const {
        vector<pair<KeyType, ValueType>> entries;
        int safe_start = std::max(0, start_box);
        int safe_end = std::min(end_box, static_cast<int>(boxes.size()) - 1);
        for (int i = safe_start; i <= safe_end; i++) {
            auto box_entries = boxes[i]->getEntries();
            entries.insert(entries.end(), box_entries.begin(), box_entries.end());
        }
        return entries;
    }

    vector<Box<KeyType, ValueType>> getPreservedBoxes(int start_box, int end_box) const {
        vector<Box<KeyType, ValueType>> preserved;
        int safe_start = std::max(0, start_box);
        int safe_end = std::min(end_box, static_cast<int>(boxes.size()) - 1);
        for (int i = safe_start; i <= safe_end; i++) {
            preserved.push_back(boxes[i]);
        }
        return preserved;
    }

    KeyType getBoxLower(int box_index) const {
        return lower_bound + box_index * box_key_range;
    }

    KeyType getBoxUpper(int box_index) const {
        KeyType candidate = lower_bound + (box_index + 1) * box_key_range;
        return candidate > upper_bound ? upper_bound : candidate;
    }

    KeyType getLowerBound() const { return lower_bound; }
    KeyType getUpperBound() const { return upper_bound; }
    size_t getBoxKeyRange() const { return box_key_range; }
    size_t getBoxCount() const { return boxes.size(); }

    vector<pair<KeyType, ValueType>> getAllEntries() const {
        vector<pair<KeyType, ValueType>> entries;
        for (const auto* box : boxes) {
            if (box != nullptr) {
                vector<pair<KeyType, ValueType>> be = box->getEntries();
                entries.insert(entries.end(), be.begin(), be.end());
            }
        }
        return entries;
    }
};

template <typename KeyType, typename ValueType>
struct IndexStructure {
    std::vector<Segment<KeyType, ValueType>*> segments;
    std::vector<KeyType> segment_start_keys;
    std::vector<int32_t> redundantArray;
    double a, b;
    std::atomic<uint64_t> version{0};
    uint64_t structure_id;
};

template <typename KeyType, typename ValueType>
class LiBox {
private:
    double underflowThreshold;
    double overflowThreshold;
    int thread_num;

    std::unique_ptr<Box<KeyType, ValueType>> small_boundary_box_;
    std::unique_ptr<Box<KeyType, ValueType>> big_boundary_box_;

    std::atomic<IndexStructure<KeyType, ValueType>*> index_structure_;

    std::atomic<bool> global_splitting_{false};
    std::queue<int32_t> split_waiting_queue_;
    std::atomic<int32_t> splitting_segment_{-1};
    mutable std::mutex split_queue_mutex_;

    static std::atomic<uint64_t> next_structure_id_;

    std::atomic<bool> is_segment_splitting_{false};
    static ThreadLocalWaitTimingStats is_segment_splitting_insert_wait_stats_;
    static ThreadLocalWaitTimingStats is_segment_splitting_delete_wait_stats_;
    static ThreadLocalWaitTimingStats is_segment_splitting_search_wait_stats_;
    static ThreadLocalWaitTimingStats wait_for_operations_stats_;

    // Accumulated timing statistics for splitSegment phases
    static std::atomic<uint64_t> accumulated_load_time_us_;
    static std::atomic<uint64_t> accumulated_wait_time_us_;
    static std::atomic<uint64_t> accumulated_prepare_time_us_;
    static std::atomic<uint64_t> accumulated_keys_time_us_;
    static std::atomic<uint64_t> accumulated_calculate_time_us_;
    static std::atomic<uint64_t> accumulated_toStruct_time_us_;
    static std::atomic<uint64_t> accumulated_create_time_us_;
    static std::atomic<uint64_t> accumulated_populate_time_us_;
    static std::atomic<uint64_t> accumulated_replace_time_us_;
    static std::atomic<uint64_t> accumulated_cleanup_time_us_;
    static std::atomic<uint64_t> accumulated_unmark_time_us_;
    static std::atomic<uint64_t> accumulated_left_seg_time_us_;
    static std::atomic<uint64_t> accumulated_right_seg_time_us_;
    static std::atomic<uint64_t> total_split_operations_;
    static std::atomic<uint64_t> total_entries_processed_;
    static std::atomic<uint64_t> total_segments_created_;
    static std::atomic<uint64_t> total_merged_entries_size_;

    static void initializeTimingStats(int thread_num) {
        // Re-initialize the static timing stats with the correct thread count
        is_segment_splitting_insert_wait_stats_ = ThreadLocalWaitTimingStats(thread_num, "LiBox is_segment_splitting (insert)");
        is_segment_splitting_delete_wait_stats_ = ThreadLocalWaitTimingStats(thread_num, "LiBox is_segment_splitting (delete)");
        is_segment_splitting_search_wait_stats_ = ThreadLocalWaitTimingStats(thread_num, "LiBox is_segment_splitting (search)");
        splitting_flag_wait_stats = ThreadLocalWaitTimingStats(thread_num, "Segment splitting_flag");
        wait_for_operations_stats_ = ThreadLocalWaitTimingStats(thread_num, "Segment wait_for_operations");
    }

    struct DeletionInfo {
        IndexStructure<KeyType, ValueType>* structure;
        int32_t replaced_segment_index;
    };

    class EpochBasedReclamation {
    private:
        struct alignas(128) ThreadSpecificEBRInfo {
            std::atomic<uint32_t> current_epoch_{0};
            uint32_t previously_accessed_epoch_{0};
            std::atomic<bool> doing_operation_{false};

            std::array<std::vector<DeletionInfo>, 3> free_lists_;

            char padding_[128 - sizeof(current_epoch_) - sizeof(previously_accessed_epoch_)
                             - sizeof(doing_operation_) - sizeof(free_lists_)];

            ThreadSpecificEBRInfo() = default;

            ThreadSpecificEBRInfo(const ThreadSpecificEBRInfo&) = delete;
            ThreadSpecificEBRInfo& operator=(const ThreadSpecificEBRInfo&) = delete;
            ThreadSpecificEBRInfo(ThreadSpecificEBRInfo&&) = delete;
            ThreadSpecificEBRInfo& operator=(ThreadSpecificEBRInfo&&) = delete;

            ~ThreadSpecificEBRInfo() {
                for (uint32_t i = 0; i < 3; ++i) {
                    free_for_epoch(i);
                }
            }

            void schedule_for_deletion(IndexStructure<KeyType, ValueType>* structure,
                                     int32_t replaced_seg_idx, uint32_t epoch) {
                if (structure) {
                    free_lists_[epoch % 3].emplace_back(DeletionInfo{structure, replaced_seg_idx});
                }
            }

            void free_for_epoch(uint32_t epoch) {
                std::vector<DeletionInfo>& free_list = free_lists_[epoch % 3];
                for (const auto& info : free_list) {
                    if (info.structure) {
                        delete_index_structure_safely(info.structure, info.replaced_segment_index);
                    }
                }
                free_list.clear();
            }

            uint32_t get_current_epoch() const {
                return current_epoch_.load(std::memory_order_acquire);
            }

            bool is_doing_operation() const {
                return doing_operation_.load(std::memory_order_acquire);
            }

        private:
            void delete_index_structure_safely(IndexStructure<KeyType, ValueType>* structure,
                                              int32_t replaced_segment_index) {
                if (!structure) return;

                if (replaced_segment_index >= 0 &&
                    replaced_segment_index < static_cast<int32_t>(structure->segments.size())) {
                    delete structure->segments[replaced_segment_index];
                }
                delete structure;
            }
        };

        std::unique_ptr<ThreadSpecificEBRInfo[]> thread_infos_;
        int max_threads_;
        std::atomic<uint32_t> global_epoch_{0};

        EpochBasedReclamation() = delete;

    public:
        explicit EpochBasedReclamation(int max_threads)
            : max_threads_(max_threads) {
            thread_infos_ = std::make_unique<ThreadSpecificEBRInfo[]>(max_threads);
        }

        ~EpochBasedReclamation() = default;

        void enter_critical_section(int thread_id) {
#ifdef LOCK_EBR
            thread_infos_[thread_id].current_epoch_.fetch_add(1, std::memory_order_acq_rel);
            thread_infos_[thread_id].doing_operation_.store(true, std::memory_order_release);
#endif
        }

        void leave_critical_section(int thread_id) {
#ifdef LOCK_EBR
            thread_infos_[thread_id].doing_operation_.store(false, std::memory_order_release);
#endif
        }

        void schedule_for_deletion(IndexStructure<KeyType, ValueType>* structure,
                                 int32_t replaced_segment_index = -1) {
            if (!structure) return;

            int thread_id = ThreadIdManager::get_thread_id();

            uint32_t current_global_epoch = global_epoch_.load(std::memory_order_acquire);
            save_epoch_snapshot_to_all_threads(current_global_epoch);

            wait_for_all_threads_safe();

            thread_infos_[thread_id].schedule_for_deletion(structure, replaced_segment_index, current_global_epoch);

            cleanup_old_epochs(current_global_epoch);

            global_epoch_.fetch_add(1, std::memory_order_acq_rel);
        }

        uint32_t get_global_epoch() const {
            return global_epoch_.load(std::memory_order_acquire);
        }

        uint32_t get_thread_epoch(int thread_id) const {
            if (thread_id >= 0 && thread_id < max_threads_) {
                return thread_infos_[thread_id].get_current_epoch();
            }
            return 0;
        }

        bool is_thread_active(int thread_id) const {
            if (thread_id >= 0 && thread_id < max_threads_) {
                return thread_infos_[thread_id].is_doing_operation();
            }
            return false;
        }

    private:
        void save_epoch_snapshot_to_all_threads(uint32_t epoch) {
            for (int i = 0; i < max_threads_; ++i) {
                thread_infos_[i].previously_accessed_epoch_ = thread_infos_[i].current_epoch_.load(std::memory_order_acquire);
            }
        }

        void wait_for_all_threads_safe() {
            for (int thread_idx = 0; thread_idx < max_threads_; ++thread_idx) {
                ThreadSpecificEBRInfo& info = thread_infos_[thread_idx];

                while (true) {
                    if (!info.doing_operation_.load(std::memory_order_acquire)) {
                        break;
                    }

                    if (info.current_epoch_.load(std::memory_order_acquire) > info.previously_accessed_epoch_) {
                        break;
                    }

                    std::this_thread::yield();
                }
            }
        }

        void cleanup_old_epochs(uint32_t current_epoch) {
            if (current_epoch >= 3) {
                uint32_t old_epoch = current_epoch - 3;

                for (int i = 0; i < max_threads_; ++i) {
                    thread_infos_[i].free_for_epoch(old_epoch);
                }
            }
        }
    };

    EpochBasedReclamation ebr_;

    class EpochGuard {
    private:
        EpochBasedReclamation* ebr_;
        int thread_id_;

    public:
        explicit EpochGuard(EpochBasedReclamation* ebr)
            : ebr_(ebr), thread_id_(ThreadIdManager::get_thread_id()) {
            ebr_->enter_critical_section(thread_id_);
        }

        ~EpochGuard() {
            ebr_->leave_critical_section(thread_id_);
        }

        EpochGuard(const EpochGuard&) = delete;
        EpochGuard& operator=(const EpochGuard&) = delete;
        EpochGuard(EpochGuard&&) = delete;
        EpochGuard& operator=(EpochGuard&&) = delete;

        int get_thread_id() const {
            return thread_id_;
        }

        EpochBasedReclamation* get_ebr() const {
            return ebr_;
        }
    };

    uint64_t generateNewStructureId() {
        return next_structure_id_.fetch_add(1);
    }

    bool is_segment_splitting(int32_t seg_index) const {
        return splitting_segment_.load(std::memory_order_acquire) == seg_index;
    }

    bool mark_segment_splitting(int32_t seg_index) {
        int32_t expected = -1;
        return splitting_segment_.compare_exchange_strong(
            expected, seg_index,
            std::memory_order_acq_rel,
            std::memory_order_acquire);
    }

    void unmark_segment_splitting(int32_t seg_index) {
        assert(splitting_segment_.load() == seg_index);
        splitting_segment_.store(-1, std::memory_order_release);
    }

    InsertResult insertToBoundaryBox(KeyType key, ValueType value, int32_t box_type) {
        auto& boundary_box = (box_type == BELOW_LOWER_BOUND) ?
                            small_boundary_box_ : big_boundary_box_;
        if (!boundary_box) {
            boundary_box = std::make_unique<Box<KeyType, ValueType>>();
        }
        InsertResult result = boundary_box->insertKeyValue(key, value);
        if (result.status == InsertStatus::SUCCESS) {
            size_t current_count = boundary_box->getTotalCount();
            if (current_count >= maxKey) {
                // Trigger upgrade to segment
                cout << "Boundary box reached max capacity, consider upgrading to segment." << endl;
                throw std::logic_error("Not implemented: upgradeBoundaryBoxToSegment");
                // upgradeBoundaryBoxToSegment(box_type == BELOW_LOWER_BOUND);
            }
        }
        return result;
    }

    SearchResult<KeyType, ValueType> searchInBoundaryBox(KeyType key, int32_t box_type) {
        const std::unique_ptr<Box<KeyType, ValueType>>& boundary_box =
            (box_type == BELOW_LOWER_BOUND) ? small_boundary_box_ : big_boundary_box_;
        if (!boundary_box) {
            return {SearchStatus::NOT_FOUND, ValueType{}};
        }
        return boundary_box->searchKey(key);
    }

    DeleteResult deleteFromBoundaryBox(KeyType key, int32_t box_type) {
        const std::unique_ptr<Box<KeyType, ValueType>>& boundary_box =
            (box_type == BELOW_LOWER_BOUND) ? small_boundary_box_ : big_boundary_box_;
        if (!boundary_box) {
            return {DeleteStatus::NOT_FOUND, false};
        }
        DeleteResult result = boundary_box->deleteKey(key);
        return result;
    }
public:
    LiBox(double uThreshold, double oThreshold, int thread_num)
        : underflowThreshold(uThreshold),
          overflowThreshold(oThreshold),
          thread_num(thread_num),
          ebr_(thread_num) {
        ThreadIdManager::initialize(thread_num);
        initializeTimingStats(thread_num);
        auto* initial = new IndexStructure<KeyType, ValueType>();
        initial->structure_id = generateNewStructureId();
        index_structure_.store(initial);
    }

    ~LiBox() {
        auto* structure = index_structure_.load();
        for (auto* seg : structure->segments) {
            delete seg;
        }
        delete structure;
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) {
        InsertResult result;
        int retry_count = 0;
        int32_t seg_index = -1;

        retry_insert:
        {
            EpochGuard guard(&ebr_);
            auto* structure = index_structure_.load(std::memory_order_acquire);
            seg_index = searchIndex(structure, key);
            if (seg_index < 0) {
                cout << "Inserting into boundary box for key: " << key << endl;
                return insertToBoundaryBox(key, value, seg_index);
            }
            if (index_structure_.load()->structure_id != structure->structure_id) {
                exponential_backoff(retry_count++);
                goto retry_insert;
            }

            result = structure->segments[seg_index]->insertKeyValue(key, value);
            if (result.status == InsertStatus::SUCCESS) {
                return result;
            }
        }

        if (result.status == InsertStatus::FULL) {
            if (!is_segment_splitting(seg_index) && splitting_segment_.load(std::memory_order_acquire) == -1) {
                if (mark_segment_splitting(seg_index)) {
                    is_segment_splitting_.store(true, std::memory_order_release);
                    splitSegment(seg_index, result.box_index);
                }else{
                    exponential_backoff(retry_count++);
                    goto retry_insert;
                }
            }
            // If multiple splits are happening, they need to be added to the queue,
            // and the operations on that segment should be blocked immediately.
            // if (splitting_segment_.load(std::memory_order_acquire) != -1) {
            //     split_waiting_queue_.push(seg_index);
            // }
            goto retry_insert;
        } else if (result.status == InsertStatus::SPLIT ||
                   result.status == InsertStatus::OUT_OF_RANGE) {
            auto wait_start = std::chrono::high_resolution_clock::now();
            is_segment_splitting_.wait(false, std::memory_order_acquire);
            auto wait_end = std::chrono::high_resolution_clock::now();
            auto wait_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(wait_end - wait_start).count();
            is_segment_splitting_insert_wait_stats_.record_wait(wait_duration);
            goto retry_insert;
        }
        return result;
    }

    DeleteResult deleteKey(KeyType key) {
        EpochGuard guard(&ebr_);
        int retry_count = 0;

    retry_delete:
        auto* structure = index_structure_.load(std::memory_order_acquire);
        int32_t seg_index = searchIndex(structure, key);
        if (seg_index < 0) {
            cout << "Deleting from boundary box for key: " << key << endl;
            return deleteFromBoundaryBox(key, seg_index);
        }
        if (index_structure_.load()->structure_id != structure->structure_id) {
            exponential_backoff(retry_count++);
            goto retry_delete;
        }

        DeleteResult ret = structure->segments[seg_index]->deleteKey(key);
        if (ret.status == DeleteStatus::SPLIT || ret.status == DeleteStatus::OUT_OF_RANGE) {
            auto wait_start = std::chrono::high_resolution_clock::now();
            is_segment_splitting_.wait(false, std::memory_order_acquire);
            auto wait_end = std::chrono::high_resolution_clock::now();
            auto wait_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(wait_end - wait_start).count();
            is_segment_splitting_delete_wait_stats_.record_wait(wait_duration);
            goto retry_delete;
        }

        return ret;
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        EpochGuard guard(&ebr_);
        int retry_count = 0;

    retry_search:
        auto* structure = index_structure_.load(std::memory_order_acquire);
        int32_t seg_index = searchIndex(structure, key);
        if (seg_index < 0) {
            cout << "Searching in boundary box for key: " << key << endl;
            return searchInBoundaryBox(key, seg_index);
        }
        if (index_structure_.load()->structure_id != structure->structure_id) {
            exponential_backoff(retry_count++);
            goto retry_search;
        }

        SearchResult<KeyType, ValueType> ret = structure->segments[seg_index]->searchKey(key);
        if (ret.status == SearchStatus::SPLIT || ret.status == SearchStatus::OUT_OF_RANGE) {
            auto wait_start = std::chrono::high_resolution_clock::now();
            is_segment_splitting_.wait(false, std::memory_order_acquire);
            auto wait_end = std::chrono::high_resolution_clock::now();
            auto wait_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(wait_end - wait_start).count();
            is_segment_splitting_search_wait_stats_.record_wait(wait_duration);
            goto retry_search;
        }

        return ret;
    }

    void populateSegmentsSerial(const vector<pair<KeyType, ValueType>>& mergedEntries,
                        vector<Segment<KeyType, ValueType>*>& new_segments) {
        size_t current_seg = 0;
        for (const auto& entry : mergedEntries) {
            KeyType key = entry.first;
            ValueType value = entry.second;
            if (key >= new_segments[current_seg]->getUpperBound()) {
                current_seg++;
            }
            new_segments[current_seg]->insertKeyValue(key, value);
        }
    }

    void splitSegment(int32_t seg_index, int box_index) {
        auto split_start = std::chrono::high_resolution_clock::now();
        cout << "splitting segment " << seg_index << " due to overflow in box " << box_index << endl;

        auto t1 = std::chrono::high_resolution_clock::now();
        if (!global_splitting_.exchange(true)) {
            auto* current = index_structure_.load();
            auto* segment = current->segments[seg_index];
            cout << "num boxes: " << segment->boxes.size() << endl;
            auto t2 = std::chrono::high_resolution_clock::now();

            auto wait_start = std::chrono::high_resolution_clock::now();
            segment->wait_for_operations();
            auto wait_end = std::chrono::high_resolution_clock::now();
            auto wait_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(wait_end - wait_start).count();
            wait_for_operations_stats_.record_wait(wait_duration);
            auto t3 = std::chrono::high_resolution_clock::now();

            // Calculate the range of boxes to process like small-split-libox.h
            int left_count = NUM_BOXES_TO_LOOK;
            int right_count = NUM_BOXES_TO_LOOK;
            size_t numBoxes = segment->getBoxCount();
            int merge_start = std::max(0, box_index - left_count);
            int merge_end = std::min(static_cast<int>(numBoxes) - 1, box_index + right_count);

            auto mergedEntries = segment->prepare_for_split_stage1(merge_start, merge_end);
            auto t4 = std::chrono::high_resolution_clock::now();

            size_t merged_entries_size = mergedEntries.size();

            if (mergedEntries.empty()) {
                unmark_segment_splitting(seg_index);
                global_splitting_.store(false);
                is_segment_splitting_.store(false, std::memory_order_release);
                is_segment_splitting_.notify_all();
                return;
            }

            vector<KeyType> keys;
            keys.reserve(mergedEntries.size());
            for (const auto& entry : mergedEntries) {
                keys.push_back(entry.first);
            }
            auto t5 = std::chrono::high_resolution_clock::now();

            KeyType merged_lower = segment->getBoxLower(merge_start);
            KeyType merged_upper = segment->getBoxUpper(merge_end);

            std::vector<keySegment<KeyType>> keysegments =
                calculateSegments(keys, overflowThreshold, underflowThreshold, 15, merged_lower, merged_upper);
            auto t6 = std::chrono::high_resolution_clock::now();

            std::vector<StructSegment<KeyType>> final_segments = toStructSegment(keysegments);
            auto t7 = std::chrono::high_resolution_clock::now(); // create phase

            std::vector<Segment<KeyType, ValueType>*> new_segments;
            std::vector<KeyType> new_segment_start_keys;

            // Declare timing variables at function scope
            std::chrono::high_resolution_clock::time_point t_left_start, t_left_end;
            std::chrono::high_resolution_clock::time_point t_right_start, t_right_end;

            // Create left segment if there are boxes before the merge range
            t_left_start = std::chrono::high_resolution_clock::now(); // left_seg phase start
            if (merge_start > 0) {

                KeyType left_lower = segment->getLowerBound();
                KeyType left_upper = segment->getBoxUpper(merge_start - 1);
                auto* left_segment = new Segment<KeyType, ValueType>(
                    left_lower, left_upper, segment->getBoxKeyRange(), thread_num
                );

                // Copy box pointers directly from the original segment to the left segment
                left_segment->boxes.resize(merge_start);
                std::copy(segment->boxes.begin(), segment->boxes.begin() + merge_start, left_segment->boxes.begin());

                new_segments.push_back(left_segment);
                new_segment_start_keys.push_back(left_lower);

            }
            t_left_end = std::chrono::high_resolution_clock::now(); // left_seg phase end

            // Process the merged entries to create new segments
            std::vector<Segment<KeyType, ValueType>*> merged_segments;
            merged_segments.reserve(final_segments.size());
            for (const auto& struct_seg : final_segments) {
                auto* new_seg = new Segment<KeyType, ValueType>(
                    struct_seg.seg_lower,
                    struct_seg.seg_upper,
                    struct_seg.box_range,
                    thread_num
                );
                merged_segments.push_back(new_seg);
                new_segments.push_back(new_seg);
                new_segment_start_keys.push_back(struct_seg.seg_lower);
            }
            auto t8 = std::chrono::high_resolution_clock::now(); // populate phase

            size_t new_segments_size = new_segments.size();

            // Populate only the merged segments (not left/right segments)
            populateSegmentsSerial(mergedEntries, merged_segments);

            // Create right segment if there are boxes after the merge range
            if (merge_end < static_cast<int>(numBoxes) - 1) {
                t_right_start = std::chrono::high_resolution_clock::now(); // right_seg phase start

                KeyType right_lower = segment->getBoxLower(merge_end + 1);
                KeyType right_upper = segment->getUpperBound();
                auto* right_segment = new Segment<KeyType, ValueType>(
                    right_lower, right_upper, segment->getBoxKeyRange(), thread_num
                );

                // Copy box pointers directly from the original segment to the right segment
                int right_box_count = numBoxes - (merge_end + 1);
                right_segment->boxes.resize(right_box_count);
                std::copy(segment->boxes.begin() + merge_end + 1, segment->boxes.end(), right_segment->boxes.begin());

                new_segments.push_back(right_segment);
                new_segment_start_keys.push_back(right_lower);

                t_right_end = std::chrono::high_resolution_clock::now(); // right_seg phase end
            }
            auto t9 = std::chrono::high_resolution_clock::now();

            atomicReplaceIndexStructure(seg_index, new_segments, split_start);
            auto t10 = std::chrono::high_resolution_clock::now();

            // Set pointers to nullptr for boxes that are now owned by left_segment and right_segment
            // to avoid double-free in the destructor
            for (int i = 0; i < merge_start; i++) {
                segment->boxes[i] = nullptr;
            }
            for (int i = merge_end + 1; i < static_cast<int>(numBoxes); i++) {
                segment->boxes[i] = nullptr;
            }
            delete segment;
            is_segment_splitting_.store(false, std::memory_order_release);
            is_segment_splitting_.notify_all();
            auto t11 = std::chrono::high_resolution_clock::now();

            unmark_segment_splitting(seg_index);
            global_splitting_.store(false);
            auto t12 = std::chrono::high_resolution_clock::now();

            // Calculate timing breakdown
            // Calculate left and right segment timing
            auto duration_left_seg = std::chrono::duration_cast<std::chrono::microseconds>(t_left_end - t_left_start).count();
            auto duration_right_seg = std::chrono::duration_cast<std::chrono::microseconds>(t_right_end - t_right_start).count();
            auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
            auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
            auto duration4 = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
            auto duration5 = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
            auto duration6 = std::chrono::duration_cast<std::chrono::microseconds>(t7 - t6).count();
            auto duration7 = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count()
                - duration_left_seg;
            auto duration8 = std::chrono::duration_cast<std::chrono::microseconds>(t9 - t8).count()
                - duration_right_seg;
            auto duration9 = std::chrono::duration_cast<std::chrono::microseconds>(t10 - t9).count();
            auto duration10 = std::chrono::duration_cast<std::chrono::microseconds>(t11 - t10).count();
            auto duration11 = std::chrono::duration_cast<std::chrono::microseconds>(t12 - t11).count();

            // Calculate total time and normalized time per 100k entries
            auto total_time_us = duration1 + duration2 + duration3 + duration4 + duration5 +
                                duration6 + duration7 + duration8 + duration9 + duration10 + duration11 +
                                duration_left_seg + duration_right_seg;
            double normalized_time_per_100k = (merged_entries_size > 0) ?
                (static_cast<double>(total_time_us) * 100000.0 / merged_entries_size) : 0.0;

            // Accumulate timing statistics
            accumulated_load_time_us_.fetch_add(duration1, std::memory_order_relaxed);
            accumulated_wait_time_us_.fetch_add(duration2, std::memory_order_relaxed);
            accumulated_prepare_time_us_.fetch_add(duration3, std::memory_order_relaxed);
            accumulated_keys_time_us_.fetch_add(duration4, std::memory_order_relaxed);
            accumulated_calculate_time_us_.fetch_add(duration5, std::memory_order_relaxed);
            accumulated_toStruct_time_us_.fetch_add(duration6, std::memory_order_relaxed);
            accumulated_create_time_us_.fetch_add(duration7, std::memory_order_relaxed);
            accumulated_populate_time_us_.fetch_add(duration8, std::memory_order_relaxed);
            accumulated_replace_time_us_.fetch_add(duration9, std::memory_order_relaxed);
            accumulated_cleanup_time_us_.fetch_add(duration10, std::memory_order_relaxed);
            accumulated_unmark_time_us_.fetch_add(duration11, std::memory_order_relaxed);
            accumulated_left_seg_time_us_.fetch_add(duration_left_seg, std::memory_order_relaxed);
            accumulated_right_seg_time_us_.fetch_add(duration_right_seg, std::memory_order_relaxed);

            // Accumulate other statistics
            total_split_operations_.fetch_add(1, std::memory_order_relaxed);
            total_entries_processed_.fetch_add(merged_entries_size, std::memory_order_relaxed);
        total_merged_entries_size_.fetch_add(merged_entries_size, std::memory_order_relaxed);
            total_segments_created_.fetch_add(new_segments_size, std::memory_order_relaxed);
        }

        auto split_end = std::chrono::high_resolution_clock::now();
        auto split_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(split_end - split_start).count();
        split_segment_total_stats.record_wait(split_duration);
    }

    void buildSearchIndex(IndexStructure<KeyType, ValueType>* structure) {
        if (structure->segment_start_keys.empty()) return;

        int64_t redundantSize = structure->segment_start_keys.size() * 90;
        structure->redundantArray.resize(redundantSize, -1);

        structure->a = static_cast<double>(redundantSize - 1) /
                      (structure->segment_start_keys.back() - structure->segment_start_keys.front());
        structure->b = -structure->a * structure->segment_start_keys.front();

        for (size_t i = 0; i < structure->segment_start_keys.size(); i++) {
            int64_t position = static_cast<int64_t>(structure->a * structure->segment_start_keys[i] + structure->b);
            if (position >= 0 && position < redundantSize) {
                structure->redundantArray[position] = i;
            }
        }

        int32_t lastValidIndex = 0;
        for (size_t i = 0; i < redundantSize; i++) {
            if (structure->redundantArray[i] == -1) {
                structure->redundantArray[i] = lastValidIndex;
            } else {
                lastValidIndex = structure->redundantArray[i];
            }
        }
    }

    int32_t searchIndex(IndexStructure<KeyType, ValueType>* structure, KeyType key) {
        auto& segment_start_keys = structure->segment_start_keys;
        auto& redundantArray = structure->redundantArray;
        double a = structure->a;
        double b = structure->b;

        if (key < segment_start_keys.front()) {
            return BELOW_LOWER_BOUND;
        } else if (key >= segment_start_keys.back()) {
            return ABOVE_UPPER_BOUND;
        }

        int64_t position = static_cast<int64_t>(a * key + b);
        int32_t estimatedIndex = redundantArray[position];

        if (segment_start_keys[estimatedIndex] <= key &&
            segment_start_keys[estimatedIndex + 1] > key) {
            return estimatedIndex;
        }

        if (segment_start_keys[estimatedIndex - 1] <= key &&
            segment_start_keys[estimatedIndex] > key) {
            return estimatedIndex - 1;
        }

        if (segment_start_keys[estimatedIndex + 1] <= key &&
            segment_start_keys[estimatedIndex + 2] > key) {
            return estimatedIndex + 1;
        }

        if (segment_start_keys[estimatedIndex] < key) {
            int32_t low = estimatedIndex + 2;
            int32_t high = low;
            int32_t step = 1;

            while (high < segment_start_keys.size() && segment_start_keys[high] <= key) {
                low = high;
                step *= 2;
                high = std::min(low + step, static_cast<int32_t>(segment_start_keys.size() - 1));
            }

            while (low <= high) {
                int32_t mid = low + (high - low) / 2;
                if (segment_start_keys[mid] <= key &&
                    (mid + 1 >= segment_start_keys.size() || segment_start_keys[mid + 1] > key)) {
                    return mid;
                }
                if (segment_start_keys[mid] <= key) {
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
        } else {
            int32_t high = estimatedIndex - 2;
            int32_t low = high;
            int32_t step = 1;

            while (low > 0 && segment_start_keys[low] > key) {
                high = low;
                step *= 2;
                low = std::max(high - step, static_cast<int32_t>(0));
            }

            while (low <= high) {
                int32_t mid = low + (high - low) / 2;
                if (segment_start_keys[mid] <= key &&
                    (mid + 1 >= segment_start_keys.size() || segment_start_keys[mid + 1] > key)) {
                    return mid;
                }
                if (segment_start_keys[mid] <= key) {
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
        }
        return -1;
    }

    void atomicReplaceIndexStructure(int32_t old_seg_idx,
                                     std::vector<Segment<KeyType, ValueType>*> new_segments,
                                     std::chrono::high_resolution_clock::time_point start_time) {
        auto* current = index_structure_.load(std::memory_order_acquire);
        auto* new_structure = new IndexStructure<KeyType, ValueType>();

        for (size_t i = 0; i < current->segments.size(); ++i) {
            if (i == old_seg_idx) {
                for (auto* seg : new_segments) {
                    new_structure->segments.push_back(seg);
                    new_structure->segment_start_keys.push_back(seg->getLowerBound());
                }
            } else {
                new_structure->segments.push_back(current->segments[i]);
                new_structure->segment_start_keys.push_back(current->segment_start_keys[i]);
            }
        }
        new_structure->segment_start_keys.push_back(
            new_structure->segments.back()->getUpperBound() + 1);

        buildSearchIndex(new_structure);

        new_structure->structure_id = generateNewStructureId();
        new_structure->version.store(0);

        auto* old = index_structure_.exchange(new_structure, std::memory_order_acq_rel);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        cout << "Total split time: " << total_duration.count() << " microseconds" << endl;

        ebr_.schedule_for_deletion(old);
    }

    void loadConfigByFile(const string& config_file) {
        ifstream config(config_file);
        if (!config.is_open()) {
            throw runtime_error("Failed to open config file.");
        }

        auto* new_structure = new IndexStructure<KeyType, ValueType>();
        string line;
        while (getline(config, line)) {
            istringstream iss(line);
            std::string token = "";
            KeyType lower, upper;
            size_t box_range;

            if (getline(iss, token, ',')) {
                if constexpr (std::is_same_v<KeyType, double>) {
                    lower = std::stod(token);
                } else if constexpr (std::is_signed_v<KeyType>) {
                    lower = std::stoll(token);
                } else {
                    lower = std::stoull(token);
                }
            }
            if (getline(iss, token, ',')) {
                if constexpr (std::is_same_v<KeyType, double>) {
                    upper = std::stod(token);
                } else if constexpr (std::is_signed_v<KeyType>) {
                    upper = std::stoll(token);
                } else {
                    upper = std::stoull(token);
                }
            }
            if (getline(iss, token)) {
                box_range = stoul(token);
            }

            auto* seg = new Segment<KeyType, ValueType>(lower, upper, box_range, thread_num);
            new_structure->segments.push_back(seg);
            new_structure->segment_start_keys.push_back(lower);
        }

        if (!new_structure->segments.empty()) {
            new_structure->segment_start_keys.push_back(
                new_structure->segments.back()->getUpperBound() + 1);
        }

        buildSearchIndex(new_structure);
        new_structure->structure_id = generateNewStructureId();

        auto* old = index_structure_.exchange(new_structure);
        delete old;
    }

    void bulk_load(std::pair<KeyType, ValueType>* key_value, size_t num) {
        size_t inserted = 0;
        omp_set_num_threads(thread_num);
#pragma omp parallel for reduction(+ : inserted)
        for (int i = 0; i < num; ++i) {
            KeyType key = key_value[i].first;
            ValueType value = key_value[i].second;
            if (insertKeyValue(key, value).status == InsertStatus::SUCCESS) {
                inserted++;
            }
        }
        std::cout << "bulk loading finished! total num: " << num << ", inserted " << inserted
                  << " keys \n";
    }

    const char* insertStatusToString(InsertStatus status) {
        switch (status) {
            case InsertStatus::SUCCESS: return "SUCCESS";
            case InsertStatus::FULL: return "FULL";
            case InsertStatus::SPLIT: return "SPLIT";
            default: return "UNKNOWN";
        }
    }

    void buildIndex(vector<KeyType>* file_keys) {
        int keys_size = file_keys->size();
        int inserted = 0;
        omp_set_num_threads(thread_num);
        #pragma omp parallel for reduction(+ : inserted)
        for (int i = 0; i < keys_size; i++) {
            if (insertKeyValue((*file_keys)[i], 1).status == InsertStatus::SUCCESS) inserted++;
        }
        cout << "bulk loading finished, inserted " << inserted << " keys \n";
    }

    void printWaitTimingStats() {
        std::cout << "\n=== Wait Timing Statistics ===" << std::endl;
        splitting_flag_wait_stats.print_stats();
        is_segment_splitting_insert_wait_stats_.print_stats();
        is_segment_splitting_delete_wait_stats_.print_stats();
        is_segment_splitting_search_wait_stats_.print_stats();
        wait_for_operations_stats_.print_stats();
        split_segment_total_stats.print_stats();
        exponential_backoff_stats.print_stats();
        std::cout << "==============================\n" << std::endl;

        // Print accumulated splitSegment statistics
        printAccumulatedSplitStats();
    }

    void printAccumulatedSplitStats() {
        uint64_t total_ops = total_split_operations_.load(std::memory_order_relaxed);
        if (total_ops == 0) {
            std::cout << "No splitSegment operations performed." << std::endl;
            return;
        }

        uint64_t load_time = accumulated_load_time_us_.load(std::memory_order_relaxed);
        uint64_t wait_time = accumulated_wait_time_us_.load(std::memory_order_relaxed);
        uint64_t prepare_time = accumulated_prepare_time_us_.load(std::memory_order_relaxed);
        uint64_t keys_time = accumulated_keys_time_us_.load(std::memory_order_relaxed);
        uint64_t calculate_time = accumulated_calculate_time_us_.load(std::memory_order_relaxed);
        uint64_t toStruct_time = accumulated_toStruct_time_us_.load(std::memory_order_relaxed);
        uint64_t create_time = accumulated_create_time_us_.load(std::memory_order_relaxed);
        uint64_t populate_time = accumulated_populate_time_us_.load(std::memory_order_relaxed);
        uint64_t replace_time = accumulated_replace_time_us_.load(std::memory_order_relaxed);
        uint64_t cleanup_time = accumulated_cleanup_time_us_.load(std::memory_order_relaxed);
        uint64_t unmark_time = accumulated_unmark_time_us_.load(std::memory_order_relaxed);
        uint64_t left_seg_time = accumulated_left_seg_time_us_.load(std::memory_order_relaxed);
        uint64_t right_seg_time = accumulated_right_seg_time_us_.load(std::memory_order_relaxed);

        uint64_t total_entries = total_entries_processed_.load(std::memory_order_relaxed);
        uint64_t total_segments = total_segments_created_.load(std::memory_order_relaxed);
        uint64_t total_merged_entries = total_merged_entries_size_.load(std::memory_order_relaxed);

        uint64_t total_time = load_time + wait_time + prepare_time + keys_time + calculate_time +
                             toStruct_time + create_time + populate_time + replace_time + cleanup_time + unmark_time +
                             left_seg_time + right_seg_time;

        std::cout << "\n=== Accumulated SplitSegment Statistics ===" << std::endl;
        std::cout << "Total operations: " << total_ops << std::endl;
        std::cout << "Total entries processed: " << total_entries << std::endl;
        std::cout << "Total segments created: " << total_segments << std::endl;
        std::cout << "Total mergedEntries size: " << total_merged_entries << std::endl;
        std::cout << "Total time: " << total_time << "us" << std::endl;
        std::cout << "Average time per operation: " << (total_time / total_ops) << "us" << std::endl;
        std::cout << "Average entries per operation: " << (total_entries / total_ops) << std::endl;
        std::cout << "Average segments per operation: " << (total_segments / total_ops) << std::endl;
        std::cout << "Overall throughput: " << std::fixed << std::setprecision(2)
                  << (total_time > 0 ? (total_entries * 1000000.0 / total_time) : 0.0) << " entries/sec" << std::endl;
        std::cout << "Normalized time per mergedEntry: " << std::fixed << std::setprecision(3)
                  << (total_merged_entries > 0 ? (static_cast<double>(total_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;

        std::cout << "\nPhase breakdown (accumulated):" << std::endl;
        std::cout << "  load: " << load_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (load_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(load_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  wait: " << wait_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (wait_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(wait_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  prepare: " << prepare_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (prepare_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(prepare_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  keys: " << keys_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (keys_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(keys_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  calculate: " << calculate_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (calculate_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(calculate_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  toStruct: " << toStruct_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (toStruct_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(toStruct_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  create: " << create_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (create_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(create_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  populate: " << populate_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (populate_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(populate_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  replace: " << replace_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (replace_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(replace_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  cleanup: " << cleanup_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (cleanup_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(cleanup_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  unmark: " << unmark_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (unmark_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(unmark_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  left_seg: " << left_seg_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (left_seg_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(left_seg_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "  right_seg: " << right_seg_time << "us (" << std::fixed << std::setprecision(1) << (total_time > 0 ? (right_seg_time * 100.0 / total_time) : 0.0) << "%) - " << std::fixed << std::setprecision(3) << (total_merged_entries > 0 ? (static_cast<double>(right_seg_time) * 1000.0 / total_merged_entries) : 0.0) << "ns per entry" << std::endl;
        std::cout << "==========================================\n" << std::endl;
    }
};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::next_structure_id_{1};

        // Static member definitions for timing stats
    template <typename KeyType, typename ValueType>
    ThreadLocalWaitTimingStats LiBox<KeyType, ValueType>::is_segment_splitting_insert_wait_stats_(1, "LiBox is_segment_splitting (insert)");

    template <typename KeyType, typename ValueType>
    ThreadLocalWaitTimingStats LiBox<KeyType, ValueType>::is_segment_splitting_delete_wait_stats_(1, "LiBox is_segment_splitting (delete)");

    template <typename KeyType, typename ValueType>
    ThreadLocalWaitTimingStats LiBox<KeyType, ValueType>::is_segment_splitting_search_wait_stats_(1, "LiBox is_segment_splitting (search)");

    template <typename KeyType, typename ValueType>
    ThreadLocalWaitTimingStats LiBox<KeyType, ValueType>::wait_for_operations_stats_(1, "Segment wait_for_operations");

    // Static member definitions for accumulated timing stats
    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_load_time_us_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_wait_time_us_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_prepare_time_us_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_keys_time_us_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_calculate_time_us_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_toStruct_time_us_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_create_time_us_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_populate_time_us_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_replace_time_us_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_cleanup_time_us_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_unmark_time_us_{0};
    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_left_seg_time_us_{0};
    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::accumulated_right_seg_time_us_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::total_split_operations_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::total_entries_processed_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::total_segments_created_{0};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::total_merged_entries_size_{0};

    //template <typename KeyType, typename ValueType>
    //ThreadLocalWaitTimingStats LiBox<KeyType, ValueType>::splitting_flag_wait_stats(1, "Segment splitting_flag");


}
