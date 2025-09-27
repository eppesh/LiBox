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

#include <unistd.h>  // for gettid()
#include <sys/syscall.h>  // for SYS_gettid

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
#define emptySlots_between 10
#define maxKey 64

#define NUM_BOXES_TO_LOOK 3

volatile int dummy;
using namespace std;

namespace liboxns {

void exponential_backoff(int retry_count);

// Helper function to get system thread ID (the one shown in GDB)
inline pid_t get_system_thread_id() {
    return syscall(SYS_gettid);
}


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
            if (keys[candidate] == key && valid_flags[candidate] && candidate < maxSize) {
                return candidate;
            }
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

   public:
    Box() {}

    Box(const Box& other)
        : maxSize(other.maxSize),
          validSize(other.validSize),
          nearestEmptySlot(other.nearestEmptySlot),
          valid_flags(other.valid_flags) {
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
                keys = other.keys;
                keys_low = other.keys_low;
                values = other.values;
                if (!other.version_changed(version_start)) {
                    break;
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
          keys(std::move(other.keys)),
          keys_low(std::move(other.keys_low)),
          values(std::move(other.values)) {

        version_lock_.store(other.version_lock_.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
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
            keys = std::move(other.keys);
            keys_low = std::move(other.keys_low);
            values = std::move(other.values);

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
        return maxSize;
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
        }

        uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
        version_lock_.store(new_version, std::memory_order_release);
        return {found ? DeleteStatus::SUCCESS : DeleteStatus::NOT_FOUND, found};
    }

    InsertResult updateValue(size_t index, ValueType value) {
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

        values[index] = value;
        uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
        version_lock_.store(new_version, std::memory_order_release);
        return {InsertStatus::SUCCESS, -1};
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

        if (nearestEmptySlot >= maxKey) {
            uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
            version_lock_.store(new_version, std::memory_order_release);
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

        uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
        version_lock_.store(new_version, std::memory_order_release);
        return {InsertStatus::SUCCESS, -1};
    }

    size_t searchUpdateKey(KeyType key) {
        int retry_count = 0;

    retry_read:
        uint32_t start_version = version_lock_.load(std::memory_order_acquire);
        if (start_version & WRITE_LOCK_BIT) {
            exponential_backoff(retry_count++);
            goto retry_read;
        }

        size_t index = findKeyIndex(key);
        if (start_version != version_lock_.load(std::memory_order_acquire)) {
            exponential_backoff(retry_count++);
            goto retry_read;
        }

        if (index != maxKey) {
            return index;
        }
        return maxKey;
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
};

ThreadLocalWaitTimingStats splitting_flag_wait_stats(84, "Segment splitting_flag");
ThreadLocalWaitTimingStats split_segment_total_stats(84, "Total splitSegment time");

template <typename KeyType, typename ValueType>
class Segment {
private:
    size_t box_key_range;
    size_t num_threads;

    mutable ThreadLocalCounter operation_counter_;
    std::atomic_flag splitting_flag_ = ATOMIC_FLAG_INIT;

    size_t logical_box_count;
    size_t physical_box_count;

    Box<KeyType, ValueType>* first_box_ptr;
    size_t active_box_count;

public:
    int thread_id;
    KeyType lower_bound;
    KeyType upper_bound;
    std::deque<std::atomic<uint8_t>> logical_box_write_positions;
    static constexpr size_t PHYSICAL_BOXES_PER_LOGICAL = 1 + overflowCapacity;
    uint8_t insert_position = (1 << PHYSICAL_BOXES_PER_LOGICAL) - 1;
    int numBoxes;
    mutable std::atomic<bool> splitting_{false};
    std::atomic<bool> is_splitting_{false};

    Segment(KeyType lower, KeyType upper, size_t box_range, int thread_num)
        : lower_bound(lower), upper_bound(upper), box_key_range(box_range),
          operation_counter_(thread_num), num_threads(thread_num) {

        size_t total = upper - lower + 1;
        logical_box_count = total / box_range;
        if (total % box_range != 0) logical_box_count++;

        physical_box_count = logical_box_count * PHYSICAL_BOXES_PER_LOGICAL;
        numBoxes = logical_box_count;
        active_box_count = logical_box_count;

        first_box_ptr = new Box<KeyType, ValueType>[physical_box_count];

        for (size_t i = 0; i < physical_box_count; i++) {
            new (&first_box_ptr[i]) Box<KeyType, ValueType>();
        }

        logical_box_write_positions.resize(logical_box_count);
        for (size_t i = 0; i < logical_box_count; i++) {
            logical_box_write_positions[i].store(0, std::memory_order_relaxed);
        }
    }

    Segment(KeyType lower, KeyType upper, size_t box_range, int thread_num,
            Box<KeyType, ValueType>* existing_boxes, size_t logical_count, bool take_ownership = false)
        : lower_bound(lower), upper_bound(upper), box_key_range(box_range),
          operation_counter_(thread_num), num_threads(thread_num) {

        logical_box_count = logical_count;
        physical_box_count = logical_count * PHYSICAL_BOXES_PER_LOGICAL;
        numBoxes = logical_count;
        active_box_count = logical_count;
        first_box_ptr = existing_boxes;

        logical_box_write_positions.resize(logical_box_count);
        for (size_t i = 0; i < logical_box_count; i++) {
            logical_box_write_positions[i].store(0, std::memory_order_relaxed);
        }
    }

    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;

    Segment(Segment&& other) noexcept
        : lower_bound(other.lower_bound),
          upper_bound(other.upper_bound),
          box_key_range(other.box_key_range),
          logical_box_count(other.logical_box_count),
          physical_box_count(other.physical_box_count),
          numBoxes(other.numBoxes),
          active_box_count(other.active_box_count),
          operation_counter_(std::move(other.operation_counter_)),
          first_box_ptr(other.first_box_ptr),
          logical_box_write_positions(std::move(other.logical_box_write_positions)) {
        splitting_.store(other.splitting_.load());
    }

    Segment& operator=(Segment&& other) noexcept {
        if (this != &other) {
            lower_bound = other.lower_bound;
            upper_bound = other.upper_bound;
            box_key_range = other.box_key_range;
            logical_box_count = other.logical_box_count;
            physical_box_count = other.physical_box_count;
            numBoxes = other.numBoxes;
            active_box_count = other.active_box_count;
            operation_counter_ = std::move(other.operation_counter_);
            first_box_ptr = other.first_box_ptr;
            logical_box_write_positions = std::move(other.logical_box_write_positions);
            splitting_.store(other.splitting_.load());
        }
        return *this;
    }

    ~Segment() {}

    size_t getLogicalBoxIndex(KeyType key) const {
        return (key - lower_bound) / box_key_range;
    }

    size_t getPhysicalBoxIndex(size_t logical_box_index, uint8_t position_offset) const {
        return logical_box_index * PHYSICAL_BOXES_PER_LOGICAL + position_offset;
    }

    bool try_mark_for_splitting() {
        bool expected = false;
        bool result = is_splitting_.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
#ifndef NDEBUG
        if (result) {
            thread_id = ThreadIdManager::get_thread_id();
            std::cout << "[DEBUG] Thread " << thread_id
                      << " (sys_tid=" << get_system_thread_id() << ")"
                      << " set is_splitting_ to TRUE in try_mark_for_splitting() for segment ("
                      << lower_bound << ", " << upper_bound << ")" << std::endl;
        }
#endif
        return result;
    }

    void unmark_splitting() {
#ifndef NDEBUG
        thread_id = ThreadIdManager::get_thread_id();
        std::cout << "[DEBUG] Thread " << thread_id
                  << " (sys_tid=" << get_system_thread_id() << ")"
                  << " set is_splitting_ to FALSE in unmark_splitting() for segment ("
                  << lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
        is_splitting_.store(false, std::memory_order_release);
        is_splitting_.notify_all();
    }

    bool is_currently_splitting() const {
        return is_splitting_.load(std::memory_order_acquire);
    }

    void wait_for_split_completion() const {
        is_splitting_.wait(true, std::memory_order_acquire);
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

        size_t logical_box_index = getLogicalBoxIndex(key);
        uint8_t current_position = logical_box_write_positions[logical_box_index].load(std::memory_order_acquire);

        uint8_t temp = insert_position;
        for (uint8_t pos = 0; pos < current_position+1; pos++) {
            size_t index = (first_box_ptr + getPhysicalBoxIndex(logical_box_index, pos))->searchUpdateKey(key);
            if (index != maxKey) {
                InsertResult result = (first_box_ptr + getPhysicalBoxIndex(logical_box_index, pos))->updateValue(index, value);
                if (result.status == InsertStatus::SUCCESS) {
                    leave();
                    return {InsertStatus::SUCCESS, static_cast<int>(logical_box_index)};
                }
                throw std::runtime_error("Unexpected update operation result");
            }
            if (!(first_box_ptr + getPhysicalBoxIndex(logical_box_index, pos))->hasEmptySlots()) {
                temp &= ~(1 << pos);
            }
        }

        uint8_t try_insert = 0;
        while (temp > 0) {
            InsertResult result = (first_box_ptr + getPhysicalBoxIndex(logical_box_index, try_insert))->insertKeyValue(key, value);
            if (result.status == InsertStatus::SUCCESS) {
                if (try_insert > logical_box_write_positions[logical_box_index].load(std::memory_order_acquire)) {
                    logical_box_write_positions[logical_box_index].store(try_insert, std::memory_order_release);
                }
                leave();
                return {InsertStatus::SUCCESS, static_cast<int>(logical_box_index)};
            }
            temp >>= 1;
            try_insert++;
        }

        leave();
        return {InsertStatus::FULL, static_cast<int>(logical_box_index)};
    }

    DeleteResult deleteKey(KeyType key) {
        if (!enter()) {
            return {DeleteStatus::SPLIT, false};
        }
        if (key < lower_bound || key > upper_bound) {
            leave();
            return {DeleteStatus::OUT_OF_RANGE, false};
        }

        size_t logical_box_index = getLogicalBoxIndex(key);
        uint8_t max_position = logical_box_write_positions[logical_box_index].load(std::memory_order_acquire);

        for (uint8_t pos = 0; pos <= max_position && pos < PHYSICAL_BOXES_PER_LOGICAL; pos++) {
            size_t physical_box_index = getPhysicalBoxIndex(logical_box_index, pos);
            DeleteResult result = (first_box_ptr + physical_box_index)->deleteKey(key);
            if (result.found) {
                leave();
                return {DeleteStatus::SUCCESS, true};
            }
        }

        leave();
        return {DeleteStatus::NOT_FOUND, false};
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        size_t logical_box_index = getLogicalBoxIndex(key);
        uint8_t max_position = logical_box_write_positions[logical_box_index].load(std::memory_order_acquire);

        for (uint8_t pos = 0; pos <= max_position && pos < PHYSICAL_BOXES_PER_LOGICAL; pos++) {
            size_t physical_box_index = getPhysicalBoxIndex(logical_box_index, pos);

            SearchResult<KeyType, ValueType> result = (first_box_ptr + physical_box_index)->searchKey(key);
            if (result.status == SearchStatus::SUCCESS) {
                return result;
            }
        }

        return {SearchStatus::NOT_FOUND, -1};
    }

    vector<pair<KeyType, ValueType>> prepare_for_split_stage1(int32_t merge_start, int32_t merge_end) {
        vector<pair<KeyType, ValueType>> mergedEntries;

        int32_t start_logical_box = std::max(0, merge_start);
        int32_t end_logical_box = std::min(merge_end, static_cast<int32_t>(logical_box_count) - 1);
        if (start_logical_box > end_logical_box) return mergedEntries;

        std::vector<size_t> box_start_positions(static_cast<size_t>(end_logical_box - start_logical_box + 2), 0);
        size_t total_size = 0;

        for (int logical_idx = start_logical_box; logical_idx <= end_logical_box; logical_idx++) {
            size_t logical_box_size = 0;
            uint8_t max_position = logical_box_write_positions[logical_idx].load(std::memory_order_acquire);

            for (uint8_t pos = 0; pos <= max_position && pos < PHYSICAL_BOXES_PER_LOGICAL; pos++) {
                size_t physical_box_index = getPhysicalBoxIndex(logical_idx, pos);
                logical_box_size += (first_box_ptr + physical_box_index)->getTotalCount();
            }

            box_start_positions[(logical_idx - start_logical_box) + 1] =
                box_start_positions[(logical_idx - start_logical_box)] + logical_box_size;
            total_size += logical_box_size;
        }

        mergedEntries.resize(total_size);

        for (int logical_idx = start_logical_box; logical_idx <= end_logical_box; logical_idx++) {
            size_t start_pos = box_start_positions[logical_idx - start_logical_box];
            size_t current_pos = start_pos;
            uint8_t max_position = logical_box_write_positions[logical_idx].load(std::memory_order_acquire);

            for (uint8_t pos = 0; pos <= max_position && pos < PHYSICAL_BOXES_PER_LOGICAL; pos++) {
                size_t physical_box_index = getPhysicalBoxIndex(logical_idx, pos);
                (first_box_ptr + physical_box_index)->getEntriesInPlace(&mergedEntries, current_pos);
                current_pos += (first_box_ptr + physical_box_index)->getTotalCount();
            }

#ifdef SORT_BOX
            size_t end_pos = box_start_positions[(logical_idx - start_logical_box) + 1];
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
    size_t getBoxCount() const { return logical_box_count; }

    vector<pair<KeyType, ValueType>> getAllEntries() const {
        vector<pair<KeyType, ValueType>> entries;
        for (size_t logical_idx = 0; logical_idx < logical_box_count; logical_idx++) {
            uint8_t max_position = logical_box_write_positions[logical_idx].load(std::memory_order_acquire);
            for (uint8_t pos = 0; pos <= max_position && pos < PHYSICAL_BOXES_PER_LOGICAL; pos++) {
                size_t physical_box_index = getPhysicalBoxIndex(logical_idx, pos);
                vector<pair<KeyType, ValueType>> box_entries = (first_box_ptr + physical_box_index)->getEntries();
                entries.insert(entries.end(), box_entries.begin(), box_entries.end());
            }
        }
        return entries;
    }

    Box<KeyType, ValueType>* getBoxPtr(size_t index) {
        size_t physical_index = getPhysicalBoxIndex(index, 0);
        return first_box_ptr + physical_index;
    }

    void resetBoxes(Box<KeyType, ValueType>* new_first_box, size_t new_logical_count) {
        first_box_ptr = new_first_box;
        logical_box_count = new_logical_count;
        physical_box_count = new_logical_count * PHYSICAL_BOXES_PER_LOGICAL;
        active_box_count = new_logical_count;
        numBoxes = new_logical_count;

        logical_box_write_positions.resize(logical_box_count);
        for (size_t i = 0; i < logical_box_count; i++) {
            logical_box_write_positions[i].store(0, std::memory_order_relaxed);
        }
    }
};

template <typename KeyType, typename ValueType>
class LiBox {
private:
    double underflowThreshold;
    double overflowThreshold;
    int thread_num;
    std::vector<Segment<KeyType, ValueType>*> segments;
    std::vector<KeyType> segment_start_keys;
    std::vector<int32_t> redundantArray;
    double a, b;

    std::unique_ptr<Box<KeyType, ValueType>> small_boundary_box_;
    std::unique_ptr<Box<KeyType, ValueType>> big_boundary_box_;

    std::atomic<int> waiting_for_critical_section_{0};

    // std::atomic<bool> global_splitting_{false};
    // std::queue<int32_t> split_waiting_queue_;
    // std::atomic<int32_t> splitting_segment_{-1};
    // mutable std::mutex split_queue_mutex_;
    std::atomic<bool> critical_section_lock_{false};

    // std::atomic<bool> is_segment_splitting_{false};
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

    void acquire_critical_section() {
        while (critical_section_lock_.exchange(true, std::memory_order_acquire)) {
            while (critical_section_lock_.load(std::memory_order_relaxed)) {
                std::this_thread::yield();
            }
        }
    }

    void release_critical_section() {
        critical_section_lock_.store(false, std::memory_order_release);
    }
public:
    LiBox(double uThreshold, double oThreshold, int thread_num)
        : underflowThreshold(uThreshold),
          overflowThreshold(oThreshold),
          thread_num(thread_num){
        ThreadIdManager::initialize(thread_num);
        initializeTimingStats(thread_num);
    }

    ~LiBox() {
        std::set<Segment<KeyType, ValueType>*> unique_segments;
        for (auto* seg : segments) {
            if (seg != nullptr) {
                unique_segments.insert(seg);
            }
        }

        for (auto* seg : unique_segments) {
            delete seg;
        }
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) {
        InsertResult result;
        int retry_count = 0;
        int32_t seg_index = -1;
        Segment<KeyType, ValueType>* target_segment = nullptr;

        retry_insert:
        {
            seg_index = searchIndex(key);
            if (seg_index < 0) {
                cout << "Inserting into boundary box for key: " << key << endl;
                return insertToBoundaryBox(key, value, seg_index);
            }

            result = segments[seg_index]->insertKeyValue(key, value);
            target_segment = segments[seg_index];
            if (result.status == InsertStatus::SUCCESS) {
                return result;
            }
        }

        if (result.status == InsertStatus::FULL) {
            if (target_segment->try_mark_for_splitting()) {
                splitSegment(target_segment, result.box_index);
            }else {
                exponential_backoff(retry_count++);
                goto retry_insert;
            }
            goto retry_insert;
        } else if (result.status == InsertStatus::SPLIT) {
            auto wait_start = std::chrono::high_resolution_clock::now();
            target_segment->wait_for_split_completion();
            auto wait_end = std::chrono::high_resolution_clock::now();
            auto wait_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(wait_end - wait_start).count();
            is_segment_splitting_insert_wait_stats_.record_wait(wait_duration);
            goto retry_insert;
        } else if (result.status == InsertStatus::OUT_OF_RANGE) {
            cout << key << " is out of range" << endl;
            throw std::runtime_error("Unexpected OUT_OF_RANGE status in insertKeyValue");
        }
        return result;
    }

    DeleteResult deleteKey(KeyType key) {
        int retry_count = 0;

    retry_delete:
        int32_t seg_index = searchIndex(key);
        if (seg_index < 0) {
            cout << "Deleting from boundary box for key: " << key << endl;
            return deleteFromBoundaryBox(key, seg_index);
        }

        DeleteResult ret = segments[seg_index]->deleteKey(key);
        Segment<KeyType, ValueType>* target_segment = segments[seg_index];
        if (ret.status == DeleteStatus::SPLIT || ret.status == DeleteStatus::OUT_OF_RANGE) {
            auto wait_start = std::chrono::high_resolution_clock::now();
            target_segment->wait_for_split_completion();
            auto wait_end = std::chrono::high_resolution_clock::now();
            auto wait_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(wait_end - wait_start).count();
            is_segment_splitting_delete_wait_stats_.record_wait(wait_duration);
            goto retry_delete;
        }

        return ret;
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        int retry_count = 0;

    retry_search:
        int32_t seg_index = searchIndex(key);
        if (seg_index < 0) {
            cout << "Searching in boundary box for key: " << key << endl;
            return searchInBoundaryBox(key, seg_index);
        }

        SearchResult<KeyType, ValueType> ret = segments[seg_index]->searchKey(key);
        Segment<KeyType, ValueType>* target_segment = segments[seg_index];
        if (ret.status == SearchStatus::SPLIT || ret.status == SearchStatus::OUT_OF_RANGE) {
            auto wait_start = std::chrono::high_resolution_clock::now();
            target_segment->wait_for_split_completion();
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

    void inPlaceReplaceSegmentWithCompression(Segment<KeyType, ValueType>* old_segment_ptr,
                                            std::vector<Segment<KeyType, ValueType>*> new_segments,
                                            std::vector<KeyType>& new_segment_start_keys) {

        CompressionStrategy strategy = planCompressionStrategy(old_segment_ptr, new_segments.size());

        if (!strategy.is_feasible) {
            std::cerr << "Error: Cannot replace segment - insufficient space" << std::endl;
            throw std::runtime_error("Cannot replace segment - insufficient space");
        }


        std::vector<int> current_old_positions;
        for (size_t i = 0; i < segments.size(); i++) {
            if (segments[i] == old_segment_ptr) {
                current_old_positions.push_back(i);
            }
        }

        if (current_old_positions.empty()) return;

        int first_old_pos = current_old_positions.front();
        int last_old_pos = current_old_positions.back();
        size_t current_old_count = current_old_positions.size();
        size_t new_count = new_segments.size();

        if (!strategy.needs_compression) {
            for (size_t i = 0; i < new_count; i++) {
                segments[current_old_positions[i]] = new_segments[i];
                segment_start_keys[current_old_positions[i]] = new_segment_start_keys[i];
            }

            for (size_t i = new_count; i < current_old_count; i++) {
                segments[current_old_positions[i]] = new_segments.back();
                segment_start_keys[current_old_positions[i]] = new_segment_start_keys.back();
            }
        } else if (strategy.forward_borrow == 0) {
            cout << "Using backward borrowing for " << strategy.backward_borrow << " segments" << endl;

            std::vector<std::pair<Segment<KeyType, ValueType>*, KeyType>> effective_segments;
            std::set<Segment<KeyType, ValueType>*> seen_segments;
            size_t spaces_found = 0;

            for (size_t i = last_old_pos + 1; i < segments.size() && spaces_found < strategy.backward_borrow; i++) {
                if (segments[i] == old_segment_ptr) {
                    spaces_found++;
                } else if (seen_segments.find(segments[i]) != seen_segments.end()) {
                    spaces_found++;
                } else {
                    effective_segments.push_back({segments[i], segment_start_keys[i]});
                    seen_segments.insert(segments[i]);
                }
            }

            for (size_t i = 0; i < new_count; i++) {
                segments[first_old_pos + i] = new_segments[i];
                segment_start_keys[first_old_pos + i] = new_segment_start_keys[i];
            }

            size_t placement_pos = first_old_pos + new_count;
            for (const auto& seg_info : effective_segments) {
                if (placement_pos < segments.size()) {
                    segments[placement_pos] = seg_info.first;
                    segment_start_keys[placement_pos] = seg_info.second;
                    placement_pos++;
                }
            }
        } else {
            std::cout << "Using bidirectional borrowing: forward=" << strategy.forward_borrow
                      << ", backward=" << strategy.backward_borrow << std::endl;

            std::vector<std::pair<Segment<KeyType, ValueType>*, KeyType>> forward_effective_segments;
            std::set<Segment<KeyType, ValueType>*> forward_seen_segments;
            size_t forward_spaces_found = 0;

            for (int i = first_old_pos - 1; i >= 0 && forward_spaces_found < strategy.forward_borrow; i--) {
                if (segments[i] == old_segment_ptr) {
                    forward_spaces_found++;
                } else if (forward_seen_segments.find(segments[i]) != forward_seen_segments.end()) {
                    forward_spaces_found++;
                } else {
                    forward_effective_segments.push_back({segments[i], segment_start_keys[i]});
                    forward_seen_segments.insert(segments[i]);
                }
            }

            std::vector<std::pair<Segment<KeyType, ValueType>*, KeyType>> backward_effective_segments;
            std::set<Segment<KeyType, ValueType>*> backward_seen_segments;
            size_t backward_spaces_found = 0;

            for (size_t i = last_old_pos + 1; i < segments.size() && backward_spaces_found < strategy.backward_borrow; i++) {
                if (segments[i] == old_segment_ptr) {
                    backward_spaces_found++;
                } else if (backward_seen_segments.find(segments[i]) != backward_seen_segments.end()) {
                    backward_spaces_found++;
                } else {
                    backward_effective_segments.push_back({segments[i], segment_start_keys[i]});
                    backward_seen_segments.insert(segments[i]);
                }
            }

            int new_start_pos = first_old_pos - strategy.forward_borrow;

            size_t placement_pos = new_start_pos;

            for (const auto& seg_info : forward_effective_segments) {
                if (placement_pos >= 0 && placement_pos < segments.size()) {
                    segments[placement_pos] = seg_info.first;
                    segment_start_keys[placement_pos] = seg_info.second;
                    placement_pos++;
                }
            }

            for (size_t i = 0; i < new_count; i++) {
                if (placement_pos < segments.size()) {
                    segments[placement_pos] = new_segments[i];
                    segment_start_keys[placement_pos] = new_segment_start_keys[i];
                    placement_pos++;
                }
            }

            for (const auto& seg_info : backward_effective_segments) {
                if (placement_pos < segments.size()) {
                    segments[placement_pos] = seg_info.first;
                    segment_start_keys[placement_pos] = seg_info.second;
                    placement_pos++;
                }
            }
        }

        buildSearchIndex();
    }

    struct CompressionStrategy {
        bool is_feasible;
        size_t forward_borrow;
        size_t backward_borrow;
        bool needs_compression;

        CompressionStrategy() : is_feasible(false), forward_borrow(0), backward_borrow(0), needs_compression(false) {}
        CompressionStrategy(bool feasible, size_t forward, size_t backward, bool compression)
            : is_feasible(feasible), forward_borrow(forward), backward_borrow(backward), needs_compression(compression) {}
    };

    size_t countBackwardAvailableSpace(int last_old_pos,
            Segment<KeyType, ValueType>* old_segment_ptr,
            const std::vector<Segment<KeyType, ValueType>*>& segments,
        size_t max_needed = SIZE_MAX) {
        size_t available_space = 0;
        std::set<Segment<KeyType, ValueType>*> seen_segments;

        for (size_t i = last_old_pos + 1; i < segments.size() && available_space < max_needed; i++) {
            if (segments[i] == old_segment_ptr) {
                available_space++;
            } else if (seen_segments.find(segments[i]) != seen_segments.end()) {
                available_space++;
            } else {
                seen_segments.insert(segments[i]);
            }
        }

        return available_space;
    }

    size_t countForwardAvailableSpace(int first_old_pos,
                Segment<KeyType, ValueType>* old_segment_ptr,
                const std::vector<Segment<KeyType, ValueType>*>& segments,
                size_t max_needed = SIZE_MAX) {
        size_t available_space = 0;
        std::set<Segment<KeyType, ValueType>*> seen_segments;

        for (int i = first_old_pos - 1; i >= 0 && available_space < max_needed; i--) {
            if (segments[i] == old_segment_ptr) {
                available_space++;
            } else if (seen_segments.find(segments[i]) != seen_segments.end()) {
                available_space++;
            } else {
                seen_segments.insert(segments[i]);
            }
        }

        return available_space;
    }

    CompressionStrategy planCompressionStrategy(Segment<KeyType, ValueType>* old_segment_ptr, size_t new_segments_count) {
        std::vector<int> current_positions;
        for (size_t i = 0; i < segments.size(); i++) {
            if (segments[i] == old_segment_ptr) {
                current_positions.push_back(i);
            }
        }

        if (current_positions.empty()) {
            return CompressionStrategy();
        }

        size_t current_available = current_positions.size();

        if (new_segments_count <= current_available) {
            return CompressionStrategy(true, 0, 0, false);
        }

        size_t extra_needed = new_segments_count - current_available;

        int last_old_pos = current_positions.back();
        size_t backward_space = countBackwardAvailableSpace(last_old_pos, old_segment_ptr, segments, extra_needed);

        if (extra_needed <= backward_space) {
            return CompressionStrategy(true, 0, extra_needed, true);
        }

        size_t remaining_needed = extra_needed - backward_space;
        int first_old_pos = current_positions.front();
        size_t forward_space = countForwardAvailableSpace(first_old_pos, old_segment_ptr, segments, remaining_needed);

        size_t total_available_space = backward_space + forward_space;

        if (extra_needed <= total_available_space) {
            return CompressionStrategy(true, remaining_needed, backward_space, true);
        }

        return CompressionStrategy();
    }

    void splitSegment(Segment<KeyType, ValueType>* segment_ptr, int box_index) {
        auto* segment = segment_ptr;

        segment->wait_for_operations();

        size_t numLogicalBoxes = segment->getBoxCount();
        std::vector<uint8_t> original_write_positions(numLogicalBoxes);
        for (size_t i = 0; i < numLogicalBoxes; i++) {
            original_write_positions[i] = segment->logical_box_write_positions[i].load(std::memory_order_acquire);
        }

        int merge_start, merge_end;
        vector<pair<KeyType, ValueType>> mergedEntries;
        std::vector<keySegment<KeyType>> keysegments;
        std::vector<StructSegment<KeyType>> final_segments;
        std::vector<Segment<KeyType, ValueType>*> merged_segments;

        {
            int left_count = NUM_BOXES_TO_LOOK;
            int right_count = NUM_BOXES_TO_LOOK;
            merge_start = std::max(0, box_index - left_count);
            merge_end = std::min(static_cast<int>(numLogicalBoxes) - 1, box_index + right_count);

            mergedEntries = segment->prepare_for_split_stage1(merge_start, merge_end);

            if (!mergedEntries.empty()) {
                vector<KeyType> keys;
                keys.reserve(mergedEntries.size());
                for (const auto& entry : mergedEntries) {
                    keys.push_back(entry.first);
                }

                KeyType merged_lower = segment->getBoxLower(merge_start);
                KeyType merged_upper = segment->getBoxUpper(merge_end);

                keysegments = calculateSegments(keys, 0.3, 0.5, 15, merged_lower, merged_upper);

                if (keysegments.size() <= 7) {
                    final_segments = toStructSegment(keysegments);

                    merged_segments.reserve(final_segments.size());
                    for (const auto& struct_seg : final_segments) {
                        auto* new_seg = new Segment<KeyType, ValueType>(
                            struct_seg.seg_lower, struct_seg.seg_upper, struct_seg.box_range, thread_num
                        );
                        merged_segments.push_back(new_seg);
                    }
                    populateSegmentsSerial(mergedEntries, merged_segments);

                    bool has_left = (merge_start > 0);
                    bool has_right = (merge_end < static_cast<int>(numLogicalBoxes) - 1);
                    size_t total_new_segments = merged_segments.size();
                    if (has_left) total_new_segments++;
                    if (has_right) total_new_segments++;

                    goto executeReplacement;
                }
                std::cout << "Local split produced too many segments (" << keysegments.size()
                << ") or insufficient space, falling back to full segment split" << std::endl;
            }

            for (auto* seg : merged_segments) {
                delete seg;
            }
            merged_segments.clear();
        }

        {
            merge_start = 0;
            merge_end = static_cast<int>(numLogicalBoxes) - 1;
            mergedEntries = segment->prepare_for_split_stage1(merge_start, merge_end);

            if (mergedEntries.empty()) {
                segment->unmark_splitting();
#ifndef NDEBUG
                segment->thread_id = ThreadIdManager::get_thread_id();
                std::cout << "[DEBUG] Thread " << segment->thread_id
                          << " (sys_tid=" << get_system_thread_id() << ")"
                          << " set is_splitting_ to FALSE in splitSegment() (empty mergedEntries) for segment ("
                          << segment->lower_bound << ", " << segment->upper_bound << ")" << std::endl;
#endif
                segment->splitting_.store(false, std::memory_order_release);
                return;
            }

            vector<KeyType> keys;
            keys.reserve(mergedEntries.size());
            for (const auto& entry : mergedEntries) {
                keys.push_back(entry.first);
            }

            KeyType merged_lower = segment->getBoxLower(merge_start);
            KeyType merged_upper = segment->getBoxUpper(merge_end);

            keysegments = calculateSegments(keys, 0.1, 0.7, 25, merged_lower, merged_upper);
            std::cout << "Full segment split with relaxed parameters produced "
                      << keysegments.size() << " segments" << std::endl;

            final_segments = toStructSegment(keysegments);

            merged_segments.reserve(final_segments.size());
            for (const auto& struct_seg : final_segments) {
                auto* new_seg = new Segment<KeyType, ValueType>(
                    struct_seg.seg_lower, struct_seg.seg_upper, struct_seg.box_range, thread_num
                );
                merged_segments.push_back(new_seg);
            }
            populateSegmentsSerial(mergedEntries, merged_segments);

            size_t total_new_segments = merged_segments.size();
        }

    executeReplacement:
        std::vector<Segment<KeyType, ValueType>*> new_segments;
        std::vector<KeyType> new_segment_start_keys;
        Box<KeyType, ValueType>* original_boxes = segment->getBoxPtr(0);
        KeyType original_upper_bound = segment->getUpperBound();

        bool is_local_split = (merge_start > 0 || merge_end < static_cast<int>(numLogicalBoxes) - 1);

        if (is_local_split) {
            bool has_left = (merge_start > 0);
            bool has_right = (merge_end < static_cast<int>(numLogicalBoxes) - 1);

            if (has_left) {
                Segment<KeyType, ValueType>* left_segment = segment;
                left_segment->upper_bound = segment->getBoxUpper(merge_start - 1);
                left_segment->resetBoxes(original_boxes, merge_start);

                for (int i = 0; i < merge_start; i++) {
                    left_segment->logical_box_write_positions[i].store(
                        original_write_positions[i], std::memory_order_relaxed);
                }

                new_segments.push_back(left_segment);
                new_segment_start_keys.push_back(left_segment->getLowerBound());
            }

            for (size_t i = 0; i < merged_segments.size(); i++) {
                new_segments.push_back(merged_segments[i]);
                new_segment_start_keys.push_back(merged_segments[i]->getLowerBound());
            }

            if (has_right) {
                KeyType right_lower = segment->getBoxLower(merge_end + 1);
                Box<KeyType, ValueType>* right_boxes = original_boxes +
                    ((merge_end + 1) * Segment<KeyType, ValueType>::PHYSICAL_BOXES_PER_LOGICAL);
                size_t right_logical_box_count = numLogicalBoxes - (merge_end + 1);

                Segment<KeyType, ValueType>* right_segment;
                if (has_left) {
                    right_segment = new Segment<KeyType, ValueType>(
                        right_lower, original_upper_bound, segment->getBoxKeyRange(), thread_num,
                        right_boxes, right_logical_box_count
                    );
                } else {
                    right_segment = segment;
                    right_segment->lower_bound = right_lower;
                    right_segment->resetBoxes(right_boxes, right_logical_box_count);
                }

                for (size_t i = 0; i < right_logical_box_count; i++) {
                    right_segment->logical_box_write_positions[i].store(
                        original_write_positions[merge_end + 1 + i], std::memory_order_relaxed);
                }

                new_segments.push_back(right_segment);
                new_segment_start_keys.push_back(right_segment->getLowerBound());
            }
        } else {
            new_segments = merged_segments;
            for (auto* seg : merged_segments) {
                new_segment_start_keys.push_back(seg->getLowerBound());
            }
        }

        acquire_critical_section();
        inPlaceReplaceSegmentWithCompression(segment, new_segments, new_segment_start_keys);
        segment->unmark_splitting();
#ifndef NDEBUG
        std::cout << "[DEBUG] Thread " << ThreadIdManager::get_thread_id()
                  << " (sys_tid=" << get_system_thread_id() << ")"
                  << " set is_splitting_ to FALSE in splitSegment() (end of function) for segment ("
                  << segment->lower_bound << ", " << segment->upper_bound << ")" << std::endl;
#endif
        segment->splitting_.store(false, std::memory_order_release);
        release_critical_section();
    }

    void insertEmptySlots(int empty_slots_between = 3) {
        if (segments.empty()) return;

        std::vector<Segment<KeyType, ValueType>*> new_segments;
        std::vector<KeyType> new_start_keys;

        for (size_t i = 0; i < segments.size(); i++) {
            new_segments.push_back(segments[i]);
            new_start_keys.push_back(segment_start_keys[i]);

            if (i < segments.size() - 1) {
                KeyType current_start = segment_start_keys[i];

                for (int j = 1; j <= empty_slots_between; j++) {
                    new_segments.push_back(segments[i]);
                    new_start_keys.push_back(current_start);
                }
            }
        }

        new_start_keys.push_back(segment_start_keys.back());

        segments = std::move(new_segments);
        segment_start_keys = std::move(new_start_keys);
    }

    void buildSearchIndex() {
        if (segment_start_keys.empty()) return;

        int64_t redundantSize = segment_start_keys.size() * 90;
        redundantArray.resize(redundantSize, -1);

        size_t memory_bytes = redundantSize * sizeof(int32_t);
        std::cout << "redundantArray size: " << redundantSize << " elements" << std::endl;
        std::cout << "redundantArray memory: " << memory_bytes << " bytes ("
                << (memory_bytes / 1024.0) << " KB, "
                << (memory_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;

        a = static_cast<double>(redundantSize - 1) /
                      (segment_start_keys.back() - segment_start_keys.front());
        b = -a * segment_start_keys.front();

        for (size_t i = 0; i < segment_start_keys.size(); i++) {
            int64_t position = static_cast<int64_t>(a * segment_start_keys[i] + b);
            if (position >= 0 && position < redundantSize) {
                redundantArray[position] = i;
            }
        }

        int32_t lastValidIndex = 0;
        for (size_t i = 0; i < redundantSize; i++) {
            if (redundantArray[i] == -1) {
                redundantArray[i] = lastValidIndex;
            } else {
                lastValidIndex = redundantArray[i];
            }
        }

    }

    int32_t searchIndex(KeyType key) {
        if (key < segment_start_keys.front()) {
            return BELOW_LOWER_BOUND;
        } else if (key >= segment_start_keys.back()) {
            return ABOVE_UPPER_BOUND;
        }

        int64_t position = static_cast<int64_t>(a * key + b);
        int32_t estimatedIndex = redundantArray[position];

        int32_t left_boundary = estimatedIndex;
        while (left_boundary > 0 &&
            segment_start_keys[left_boundary - 1] == segment_start_keys[estimatedIndex]) {
            left_boundary--;
        }

        int32_t right_boundary = estimatedIndex;
        while (right_boundary < static_cast<int32_t>(segment_start_keys.size() - 1) &&
            segment_start_keys[right_boundary + 1] == segment_start_keys[estimatedIndex]) {
            right_boundary++;
        }

        KeyType current_key = segment_start_keys[estimatedIndex];
        KeyType next_key = (right_boundary < static_cast<int32_t>(segment_start_keys.size() - 1)) ?
                        segment_start_keys[right_boundary + 1] :
                        std::numeric_limits<KeyType>::max();

        if (current_key <= key && key < next_key) {
            return estimatedIndex;
        }

        if (key < current_key) {
            if (left_boundary > 0) {
                int32_t prev_index = left_boundary - 1;
                KeyType prev_key = segment_start_keys[prev_index];
                if (prev_key <= key && key < current_key) {
                    return prev_index;
                }
            }
        } else { // key >= next_key
            if (right_boundary < static_cast<int32_t>(segment_start_keys.size() - 1)) {
                int32_t next_index = right_boundary + 1;

                int32_t next_right_boundary = next_index;
                while (next_right_boundary < static_cast<int32_t>(segment_start_keys.size() - 1) &&
                    segment_start_keys[next_right_boundary + 1] == segment_start_keys[next_index]) {
                    next_right_boundary++;
                }

                KeyType next_next_key = (next_right_boundary < static_cast<int32_t>(segment_start_keys.size() - 1)) ?
                                    segment_start_keys[next_right_boundary + 1] :
                                    std::numeric_limits<KeyType>::max();

                if (segment_start_keys[next_index] <= key && key < next_next_key) {
                    return next_index;
                }
            }
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

    void loadConfigByFile(const string& config_file) {
        ifstream config(config_file);
        if (!config.is_open()) {
            throw runtime_error("Failed to open config file.");
        }

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
            segments.push_back(seg);
            segment_start_keys.push_back(lower);
        }

        if (!segments.empty()) {
            segment_start_keys.push_back(
                segments.back()->getUpperBound() + 1);
        }

        insertEmptySlots(emptySlots_between);
        buildSearchIndex();
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
