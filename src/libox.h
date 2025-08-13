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
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "segmentation.h"

#define overflowCapacity 5
#define maxKey 64

volatile int dummy;
using namespace std;

namespace liboxns {
class OpenMPThreadLocalCounter {
private:
    struct alignas(64) ThreadCounter {
        std::atomic<uint32_t> count{0};
        char padding[60];
    };
    
    std::unique_ptr<ThreadCounter[]> thread_counters_;
    int thread_num_;
    
    int get_thread_slot() const {
        int thread_id = omp_get_thread_num();
        if (thread_id >= thread_num_) {
            thread_id = thread_id % thread_num_;
        }
        return thread_id;
    }
    
public:
    explicit OpenMPThreadLocalCounter(int thread_num) 
        : thread_num_(thread_num) {
        assert(thread_num > 0);
        
        thread_counters_ = std::make_unique<ThreadCounter[]>(thread_num);
        
        for (int i = 0; i < thread_num; i++) {
            thread_counters_[i].count.store(0, std::memory_order_relaxed);
        }
    }
    
    OpenMPThreadLocalCounter(const OpenMPThreadLocalCounter& other) 
        : thread_num_(other.thread_num_) {
        thread_counters_ = std::make_unique<ThreadCounter[]>(thread_num_);
        for (int i = 0; i < thread_num_; i++) {
            thread_counters_[i].count.store(
                other.thread_counters_[i].count.load(std::memory_order_relaxed),
                std::memory_order_relaxed
            );
        }
    }
    
    OpenMPThreadLocalCounter(OpenMPThreadLocalCounter&& other) noexcept 
        : thread_counters_(std::move(other.thread_counters_)),
          thread_num_(other.thread_num_) {
    }
    
    OpenMPThreadLocalCounter& operator=(const OpenMPThreadLocalCounter& other) {
        if (this != &other) {
            thread_num_ = other.thread_num_;
            thread_counters_ = std::make_unique<ThreadCounter[]>(thread_num_);
            for (int i = 0; i < thread_num_; i++) {
                thread_counters_[i].count.store(
                    other.thread_counters_[i].count.load(std::memory_order_relaxed),
                    std::memory_order_relaxed
                );
            }
        }
        return *this;
    }
    
    OpenMPThreadLocalCounter& operator=(OpenMPThreadLocalCounter&& other) noexcept {
        if (this != &other) {
            thread_counters_ = std::move(other.thread_counters_);
            thread_num_ = other.thread_num_;
        }
        return *this;
    }
    
    void increment() {
        int slot = get_thread_slot();
        thread_counters_[slot].count.fetch_add(1, std::memory_order_relaxed);
    }
    
    uint64_t get_total_count() const {
        uint64_t total = 0;
        for (int i = 0; i < thread_num_; i++) {
            total += thread_counters_[i].count.load(std::memory_order_relaxed);
        }
        return total;
    }
    
    std::vector<uint32_t> get_snapshot() const {
        std::vector<uint32_t> snapshot(thread_num_);
        for (int i = 0; i < thread_num_; i++) {
            snapshot[i] = thread_counters_[i].count.load(std::memory_order_acquire);
        }
        return snapshot;
    }
    
    bool changed_since_snapshot(const std::vector<uint32_t>& snapshot) const {
        if (snapshot.size() != static_cast<size_t>(thread_num_)) return true;
        
        for (int i = 0; i < thread_num_; i++) {
            if (thread_counters_[i].count.load(std::memory_order_acquire) != snapshot[i]) {
                return true;
            }
        }
        return false;
    }
    
    void print_distribution() const {
        std::cout << "OpenMP Thread Counter distribution:" << std::endl;
        for (int i = 0; i < thread_num_; i++) {
            uint32_t count = thread_counters_[i].count.load(std::memory_order_relaxed);
            if (count > 0) {
                std::cout << "Thread[" << i << "] = " << count << std::endl;
            }
        }
        std::cout << "Total: " << get_total_count() << std::endl;
    }
    
    int get_current_thread_slot() const {
        return get_thread_slot();
    }
    
    int get_thread_num() const {
        return thread_num_;
    }
};
enum class InsertStatus {
    SUCCESS,
    FULL,
    SPLIT,
    WRITING,
};

struct InsertResult {
    InsertStatus status;
    int box_index;
};

enum class DeleteStatus {
    SUCCESS,
    NOT_FOUND,
    SPLIT,
    ERROR
};

struct DeleteResult {
    DeleteStatus status;
    bool found;
};

enum class SearchStatus { SUCCESS, NOT_FOUND, ERROR, SPLIT};

template <typename KeyType, typename ValueType>
struct SearchResult {
    SearchStatus status;
    ValueType value;
};

class NoOpLock {
   public:
    void lock() {}
    void unlock() {}
};

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

    void updateValueAt_unsafe(int index, ValueType value) {
        values[index] = value;
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

    bool deleteKey_unsafe(KeyType key) {
        size_t index = findKeyIndex(key);
        if (index != maxKey) {
            valid_flags[index] = 0;
            validSize--;
            nearestEmptySlot = index < nearestEmptySlot ? index : nearestEmptySlot;
            return true;
        }
        return false;
    }

    size_t scan_optimized(KeyType key_low_bound,
                          size_t max_count,
                          pair<KeyType, ValueType>* result,
                          bool need_filter = true) const {
        if (max_count == 0 || result == nullptr || maxSize == 0) {
            return 0;
        }

        if (!need_filter) {
            return copyDataWithLimit(result, max_count);
        } else {
            return avx512_filter_keys_optimized(key_low_bound, result, max_count);
        }
    }

    size_t scan(KeyType key_low_bound,
                size_t max_count,
                pair<KeyType, ValueType>* result,
                bool need_filter = true) const {
        return scan_optimized(key_low_bound, max_count, result, need_filter);
    }

    size_t scan_load(pair<KeyType, ValueType>* result, size_t max_count) const {
        return copyDataWithLimit(result, max_count);
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

    InsertResult insert_unsafe(KeyType key, ValueType value) {      
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

    vector<pair<KeyType, ValueType>> rangeSearch(KeyType start_key, KeyType end_key) const {
        vector<pair<KeyType, ValueType>> results;
        const size_t batch_size = 8;
        size_t num_batches = (maxSize + batch_size - 1) / batch_size;
        for (size_t batch = 0; batch < num_batches; batch++) {
            size_t base_idx = batch * batch_size;

            __m512i v_keys = _mm512_load_si512(reinterpret_cast<const __m512i*>(&keys[base_idx]));
            __m512i v_start = _mm512_set1_epi64(start_key);
            __m512i v_end = _mm512_set1_epi64(end_key);

            __mmask8 mask_ge = _mm512_cmpge_epi64_mask(v_keys, v_start);
            __mmask8 mask_le = _mm512_cmple_epi64_mask(v_keys, v_end);
            __mmask8 mask_in_range = mask_ge & mask_le;

            while (mask_in_range) {
                int pos = __builtin_ctzll(mask_in_range);
                size_t actual_idx = base_idx + pos;

                if (actual_idx < maxSize) {
                    results.push_back({keys[actual_idx], values[actual_idx]});
                }

                mask_in_range &= mask_in_range - 1;
            }
        }

        return results;
    }

    SearchResult<KeyType, ValueType> search(KeyType key) const {
        size_t index = findKeyIndex(key);
        if (index != maxKey) {
            return {SearchStatus::SUCCESS, values[index]};
        }
        return {SearchStatus::NOT_FOUND, std::numeric_limits<ValueType>::max()};
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

template <typename KeyType, typename ValueType>
class Box {
   private:
    size_t maxSize = 0;
    size_t validSize = 0;
    size_t nearestEmptySlot = 0;
    bitset<maxKey> valid_flags;

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
            nearestEmptySlot = -1;
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

    BoxSearchResult findKeyOrSlot_unsafe(KeyType key, ValueType value) {
        size_t existingIndex = findKeyIndex(key);
        if (existingIndex != maxKey) {
            values[existingIndex] = value;
            return {BoxInsertResult::UPDATE, 0, existingIndex};
        }
        
        if (validSize < maxKey) {
            for (size_t i = 0; i < capacity; i++) {
                size_t existingIndex = data[i]->findKeyIndex(key);
                if (existingIndex != maxKey) {
                    data[i]->updateValueAt_unsafe(existingIndex, value);
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
                    data[i]->updateValueAt_unsafe(existingIndex, value);
                    return {BoxInsertResult::UPDATE, static_cast<size_t>(i+1), existingIndex};
                }
                if (data[i]->hasEmptySlots()) {
                    insertBox = i;
                }
            }
            if (insertBox == capacity) {
                if (capacity == overflowCapacity) {
                    return {BoxInsertResult::FULL, 0, 0};
                }
                data[capacity] = make_unique<OverflowKeyValue<KeyType, ValueType>>();
                data[capacity]->insert_unsafe(key, value);
                capacity++;
                return {BoxInsertResult::INSERT, capacity, 0};
            } else {
                data[insertBox]->insert_unsafe(key, value);
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
        size_t total = 0;
        while (true) {
            uint32_t version_start;
            if (test_write_lock(version_start)) {
                std::this_thread::yield();
                continue;
            }
            total = maxSize;
            if (!version_changed(version_start)) {
                break;
            }
        }
        for (size_t i = 0; i < capacity; i++) {
            total += data[i]->getTotalCount();
        }
        return total;
    }

    size_t scan_optimized(KeyType key_low_bound,
                        size_t max_count,
                        pair<KeyType, ValueType>* result,
                        bool need_filter = true) const {
        if (max_count == 0 || result == nullptr) {
            return 0;
        }

        size_t collected = 0;
        if (!need_filter) {
            collected = copyMainKeysWithLimit(result, max_count);
        } else {
            collected = avx512_filter_main_keys_optimized(key_low_bound, result, max_count);
        }
        if (collected < max_count) {
            for (size_t i = 0; i < capacity && collected < max_count; i++) {
                size_t box_collected = data[i]->scan_optimized(key_low_bound,
                                                                max_count - collected,
                                                                result + collected,
                                                                true // need_filter = true
                );
                collected += box_collected;
            }
        }
        return collected;
    }

    size_t scan(KeyType key_low_bound,
                size_t max_count,
                pair<KeyType, ValueType>* result,
                bool need_filter = true) const {
        return scan_optimized(key_low_bound, max_count, result, need_filter);
    }

    DeleteResult deleteKey(KeyType key) {
        if (!try_acquire_write_lock()) {
            return {DeleteStatus::SPLIT, false};
        }
        size_t index = findKeyIndex(key);
        if (index != maxKey) {
            valid_flags[index] = 0;
            validSize--;
            nearestEmptySlot = index < nearestEmptySlot ? index : nearestEmptySlot;
            release_write_lock();
            return {DeleteStatus::SUCCESS, true};
        }
        for (size_t i = 0; i < capacity; i++) {
            if (data[i] && data[i]->deleteKey(key)) {
                release_write_lock();
                return {DeleteStatus::SUCCESS, true};
            }
        }
        release_write_lock();
        return {DeleteStatus::NOT_FOUND, false};
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) { 
        if (!try_acquire_write_lock()) {
            return {InsertStatus::WRITING, -1};
        }
        BoxSearchResult ret = findKeyOrSlot(key, value); 
        InsertStatus status;
        if (ret.isUpdate == 0 || ret.isUpdate == 1) {
            status = InsertStatus::SUCCESS;
        } else {
            status = InsertStatus::FULL;
        }
        
        release_write_lock();
        return {status, -1};
    }

    InsertResult insertKeyValue_unsafe(KeyType key, ValueType value) {
        BoxSearchResult ret = findKeyOrSlot_unsafe(key, value);
        InsertStatus status;
        if (ret.isUpdate == BoxInsertResult::UPDATE || ret.isUpdate == BoxInsertResult::INSERT) {
            status = InsertStatus::SUCCESS;
        } else {
            status = InsertStatus::FULL;
        }
        
        return {status, -1};
    }

    vector<pair<KeyType, ValueType>> rangeSearch(KeyType start_key, KeyType end_key) const {
        vector<pair<KeyType, ValueType>> results;
        {
            const size_t batch_size = 8;
            size_t num_batches = (maxSize + batch_size - 1) / batch_size;

            for (size_t batch = 0; batch < num_batches; batch++) {
                size_t base_idx = batch * batch_size;

                __m512i v_keys = _mm512_load_si512(reinterpret_cast<const __m512i*>(&keys[base_idx]));
                __m512i v_start = _mm512_set1_epi64(start_key);
                __m512i v_end = _mm512_set1_epi64(end_key);

                __mmask8 mask_ge = _mm512_cmpge_epi64_mask(v_keys, v_start);
                __mmask8 mask_le = _mm512_cmple_epi64_mask(v_keys, v_end);
                __mmask8 mask_in_range = mask_ge & mask_le;

                while (mask_in_range) {
                    int pos = __builtin_ctzll(mask_in_range);
                    size_t actual_idx = base_idx + pos;

                    if (actual_idx < maxSize) {
                        results.push_back({keys[actual_idx], values[actual_idx]});
                    }

                    mask_in_range &= mask_in_range - 1;
                }
            }
        }

        for (size_t i = 0; i < capacity; i++) {
            auto box_results = data[i]->rangeSearch(start_key, end_key);
            results.insert(results.end(), box_results.begin(), box_results.end());
        }

        return results;
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) const {
        while (true) {
            uint32_t version_start;
            if (test_write_lock(version_start)) {
                std::this_thread::yield();
                continue;
            }            
            size_t index = findKeyIndex(key);
            if (index != maxKey) {
                ValueType result_value = values[index];
                
                if (!version_changed(version_start)) {
                    return {SearchStatus::SUCCESS, result_value};
                }
                continue;
            }            
            if (version_changed(version_start)) {
                continue;
            }            
            for (size_t i = 0; i < capacity; i++) {
                SearchResult<KeyType, ValueType> ret = data[i]->search(key);
                if (ret.status == SearchStatus::SUCCESS) {
                    if (!version_changed(version_start)) {
                        return ret;
                    }
                    break;
                }
            }
            if (!version_changed(version_start)) {
                return {SearchStatus::NOT_FOUND, std::numeric_limits<ValueType>::max()};
            }            
        }
    }

    size_t getmaxSize() const { 
        return maxSize; 
    }

    vector<pair<KeyType, ValueType>> getEntries() const {
        vector<pair<KeyType, ValueType>> entries;
        {
            for (size_t i = 0; i < maxSize; i++) {
                entries.push_back({keys[i], values[i]});
            }
        }
        for (int i = 0; i < capacity; i++) {
            vector<pair<KeyType, ValueType>> be = data[i]->getEntries();
            entries.insert(entries.end(), be.begin(), be.end());
        }
        return entries;
    }

    size_t scan_load(pair<KeyType, ValueType>* result, size_t max_count) const {
        size_t collected = 0;
        collected = copyMainKeysWithLimit(result, max_count);
        for (size_t i = 0; i < capacity && collected < max_count; i++) {
            size_t box_collected = data[i]->scan_load(result + collected, max_count - collected);
            collected += box_collected;
        }

        return collected;
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

template <typename KeyType, typename ValueType>
class Segment {
private:
    KeyType lower_bound;
    KeyType upper_bound;
    size_t box_key_range;
    int numBoxes;

    mutable OpenMPThreadLocalCounter modification_counter_;
    mutable std::atomic<bool> splitting_{false};
public:
    std::vector<Box<KeyType, ValueType>> boxes;

    Segment(KeyType lower, KeyType upper, size_t box_range, int thread_num)
        : lower_bound(lower), upper_bound(upper), box_key_range(box_range),
          modification_counter_(thread_num) {
        size_t total = upper - lower + 1;
        size_t box_count = total / box_range;
        if (total % box_range != 0) box_count++;
        numBoxes = box_count;
        boxes.resize(box_count);
    }

    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;

    Segment(Segment&& other) noexcept
        : lower_bound(other.lower_bound),
          upper_bound(other.upper_bound),
          box_key_range(other.box_key_range),
          numBoxes(other.numBoxes),
          modification_counter_(std::move(other.modification_counter_)),
          boxes(std::move(other.boxes)) {
        splitting_.store(other.splitting_.load());
    }

    Segment& operator=(Segment&& other) noexcept {
        if (this != &other) {
            lower_bound = other.lower_bound;
            upper_bound = other.upper_bound;
            box_key_range = other.box_key_range;
            numBoxes = other.numBoxes;
            modification_counter_ = std::move(other.modification_counter_);
            boxes = std::move(other.boxes);
            splitting_.store(other.splitting_.load());
        }
        return *this;
    }

    std::vector<uint32_t> get_modification_snapshot() const {
        return modification_counter_.get_snapshot();
    }
    
    bool modified_since_snapshot(const std::vector<uint32_t>& snapshot) const {
        return modification_counter_.changed_since_snapshot(snapshot);
    }
    
    void increment_modification_count() {
        modification_counter_.increment();
    }
    
    uint64_t get_total_modifications() const {
        return modification_counter_.get_total_count();
    }

    bool is_splitting() const {
        return splitting_.load(std::memory_order_acquire);
    }

    void set_splitting(bool splitting) {
        splitting_.store(splitting, std::memory_order_release);
    }

    InsertResult insertKeyValue(KeyType key, ValueType value, bool unsafe_mode = false) {
        if (!unsafe_mode && is_splitting()) {
            return {InsertStatus::SPLIT, -1};
        }
        size_t box_index = (key - lower_bound) / box_key_range;
        if (box_index >= boxes.size()) {
            if (unsafe_mode) {
                static std::mutex boxes_expansion_mutex;
                std::lock_guard<std::mutex> lock(boxes_expansion_mutex);
                while (box_index >= boxes.size()) {
                    boxes.emplace_back();
                }
            } else {
                return {InsertStatus::FULL, static_cast<int>(box_index)};
            }
        }
        
        InsertResult ret;
        if (unsafe_mode) {
            ret = boxes[box_index].insertKeyValue_unsafe(key, value);
        } else {
            ret = boxes[box_index].insertKeyValue(key, value);
            if (ret.status == InsertStatus::WRITING) {
                return insertKeyValue(key, value, false);
            }
        }
        if (ret.status == InsertStatus::FULL) {
            ret.box_index = static_cast<int>(box_index);
        }
        if (!unsafe_mode && ret.status == InsertStatus::SUCCESS) {
            increment_modification_count();
        }
        return ret;
    }

    DeleteResult deleteKey(KeyType key) {
        if (is_splitting()) {
            return {DeleteStatus::SPLIT, false};
        }
        size_t box_index = (key - lower_bound) / box_key_range;
        if (box_index >= boxes.size()) {
            return {DeleteStatus::ERROR, false};
        }
        DeleteResult result = boxes[box_index].deleteKey(key);
        if (result.status == DeleteStatus::SUCCESS) {
            increment_modification_count();
        }
        return result;
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) const {
        if (is_splitting()) {
            return {SearchStatus::SPLIT, std::numeric_limits<ValueType>::max()};
        }
        size_t box_index = (key - lower_bound) / box_key_range;
        if (box_index >= boxes.size()) {
            return {SearchStatus::NOT_FOUND, std::numeric_limits<ValueType>::max()};
        }
        return boxes[box_index].searchKey(key);
    }

    vector<pair<KeyType, ValueType>> rangeSearch(KeyType start_key, KeyType end_key) const {
        if (is_splitting()) {
            return {};
        }
        vector<pair<KeyType, ValueType>> all_results;
        size_t start_box = 0;
        size_t end_box = boxes.size() - 1;
        
        if (start_key > lower_bound) {
            start_box = (start_key - lower_bound) / box_key_range;
        }
        if (end_key < upper_bound) {
            end_box = min(end_box, (end_key - lower_bound) / box_key_range);
        }
        
        for (size_t i = start_box; i <= end_box && i < boxes.size(); i++) {
            auto box_results = boxes[i].rangeSearch(start_key, end_key);
            all_results.insert(all_results.end(), box_results.begin(), box_results.end());
        }
        return all_results;
    }

    size_t scan_optimized(KeyType key_low_bound,
                          size_t max_count,
                          pair<KeyType, ValueType>* result,
                          bool need_filter = true) const {
        if (is_splitting() || max_count == 0 || result == nullptr || boxes.empty()) {
            return 0;
        }
        
        size_t start_box_idx = 0;
        if (key_low_bound > lower_bound) {
            start_box_idx = (key_low_bound - lower_bound) / box_key_range;
            if (start_box_idx >= boxes.size()) {
                return 0;
            }
        }
        
        size_t collected = 0;
        for (size_t box_idx = start_box_idx; box_idx < boxes.size() && collected < max_count; box_idx++) {
            size_t remaining = max_count - collected;
            bool box_need_filter = need_filter;
            KeyType box_lower = getBoxLower(box_idx);
            
            if (key_low_bound <= box_lower) {
                box_need_filter = false;
            }
            
            if (!box_need_filter) {
                size_t box_total = boxes[box_idx].getTotalCount();
                if (box_total <= remaining) {
                    size_t box_collected = boxes[box_idx].scan_load(result + collected, remaining);
                    collected += box_collected;
                } else {
                    size_t box_collected = boxes[box_idx].scan_optimized(key_low_bound,
                                                                         remaining,
                                                                         result + collected);
                    collected += box_collected;
                    break;
                }
            } else {
                size_t box_collected = boxes[box_idx].scan_optimized(key_low_bound,
                                                                     remaining,
                                                                     result + collected);
                collected += box_collected;
                if (box_collected == 0) {
                    if (key_low_bound > getBoxUpper(box_idx)) {
                        continue;
                    } else {
                        break;
                    }
                }
            }
        }
        return collected;
    }

    size_t scan(KeyType key_low_bound,
                size_t max_count,
                pair<KeyType, ValueType>* result,
                bool need_filter = true) const {
        return scan_optimized(key_low_bound, max_count, result, need_filter);
    }

    vector<pair<KeyType, ValueType>> prepare_for_split_stage1(int32_t box_index) {
        int left_count = 3;
        int right_count = 3;
        int merge_start = std::max(0, box_index - left_count);
        int merge_end = std::min(static_cast<int>(boxes.size()) - 1, box_index + right_count);
        
        vector<pair<KeyType, ValueType>> mergedEntries;
        for (int i = merge_start; i <= merge_end; i++) {
            auto entries = boxes[i].getEntries();
            mergedEntries.insert(mergedEntries.end(), entries.begin(), entries.end());
        }
        
        std::sort(mergedEntries.begin(), mergedEntries.end(),
                 [](const pair<KeyType, ValueType>& a, const pair<KeyType, ValueType>& b) {
                     return a.first < b.first;
                 });
        return mergedEntries;
    }

    vector<pair<KeyType, ValueType>> getBoxRangeEntries(int start_box, int end_box) const {
        vector<pair<KeyType, ValueType>> entries;
        int safe_start = std::max(0, start_box);
        int safe_end = std::min(end_box, static_cast<int>(boxes.size()) - 1);
        for (int i = safe_start; i <= safe_end; i++) {
            auto box_entries = boxes[i].getEntries();
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
        for (const auto& box : boxes) {
            vector<pair<KeyType, ValueType>> be = box.getEntries();
            entries.insert(entries.end(), be.begin(), be.end());
        }
        return entries;
    }
};

template <typename KeyType, typename ValueType>
class LiBox {
private:
    double a;
    double b;
    vector<int32_t> redundantArray;
    vector<Segment<KeyType, ValueType>> segments;
    vector<KeyType> segment_start_keys;
    int underflowThreshold;
    int overflowThreshold;
    int thread_num;

    mutable std::atomic<uint32_t> global_split_version_{0};
    static constexpr uint32_t SPLIT_IN_PROGRESS = 0x80000000;
    static constexpr uint32_t VERSION_MASK = 0x7FFFFFFF;

    struct BatchScanQuery {
        KeyType start_key;
        int scan_count;
        int original_index;
    };

    struct RangeSearchResult {
        vector<pair<KeyType, ValueType>> results;
        size_t total_boxes_accessed;
        size_t total_keys_examined;
        RangeSearchResult() : total_boxes_accessed(0), total_keys_examined(0) {}
    };

    bool is_global_splitting() const {
        return (global_split_version_.load(std::memory_order_acquire) & SPLIT_IN_PROGRESS) != 0;
    }

    bool try_start_split() {
        uint32_t expected = global_split_version_.load(std::memory_order_acquire);
        while (expected & SPLIT_IN_PROGRESS) {
            std::this_thread::yield();
            expected = global_split_version_.load(std::memory_order_acquire);
        }
        
        uint32_t desired = expected | SPLIT_IN_PROGRESS;
        return global_split_version_.compare_exchange_strong(expected, desired,
                                                            std::memory_order_acq_rel,
                                                            std::memory_order_acquire);
    }
    
    void finish_split() {
        uint32_t current = global_split_version_.load(std::memory_order_relaxed);
        uint32_t new_version = ((current + 1) & VERSION_MASK);
        global_split_version_.store(new_version, std::memory_order_release);
    }

    void wait_for_operations_complete() {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    
    bool validate_segments_consistency() const {
        if (segment_start_keys.empty() || segments.empty()) {
            return segment_start_keys.size() == segments.size();
        }
        return segment_start_keys.size() == segments.size() + 1;
    }
public:
    LiBox(int uThreshold, int oThreshold, int thread_num)
        : underflowThreshold(uThreshold),
          overflowThreshold(oThreshold),
          thread_num(thread_num) {}

    ~LiBox() {
        segments.clear();
        segment_start_keys.clear();
    }

    void init(int uThreshold, int oThreshold, int threadNum) {
        underflowThreshold = uThreshold;
        overflowThreshold = oThreshold;
        thread_num = threadNum;
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) {
        if (is_global_splitting()) {
            std::this_thread::yield();
            return insertKeyValue(key, value);
        }
        int32_t num_index = searchIndex(key);
        InsertResult ret = segments[num_index].insertKeyValue(key, value);
        if (ret.status == InsertStatus::SUCCESS) {
            return ret;
        } else if (ret.status == InsertStatus::SPLIT) {
            std::this_thread::yield();
            return insertKeyValue(key, value);
        } else if (ret.status == InsertStatus::FULL) {
            int32_t box_index = ret.box_index;
            splitSegment(num_index, box_index);
            return insertKeyValue(key, value);;
        } else {
            return ret;
        }
    }

    DeleteResult deleteKey(KeyType key) {
        if (is_global_splitting()) {
            std::this_thread::yield();
            return deleteKey(key);
        }
        int32_t num_index = searchIndex(key);
        if (num_index < 0 || num_index >= static_cast<int32_t>(segments.size())) {
            return {DeleteStatus::ERROR, false};
        }
        DeleteResult ret = segments[num_index].deleteKey(key);
        if (ret.status != DeleteStatus::SPLIT) {
            return ret;
        }
        std::this_thread::yield();
        return deleteKey(key);
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        if (is_global_splitting()) {
            std::this_thread::yield();
            return searchKey(key);
        }
        int32_t num_index = searchIndex(key);
        if (num_index < 0 || num_index >= static_cast<int32_t>(segments.size())) {
            return {SearchStatus::ERROR, std::numeric_limits<ValueType>::max()};
        }
        SearchResult<KeyType, ValueType> ret = segments[num_index].searchKey(key);
        if (ret.status != SearchStatus::SPLIT) {
            return ret;
        }
        std::this_thread::yield();
        return searchKey(key);
    }

    RangeSearchResult rangeSearch(KeyType start_key, KeyType end_key) {
        RangeSearchResult result;
        if (start_key > end_key) {
            return result;
        }
        
        const int MAX_RETRIES = 10;
        int retry_count = 0;
        
        while (retry_count < MAX_RETRIES) {
            if (is_global_splitting()) {
                std::this_thread::yield();
                retry_count++;
                continue;
            }
            
            vector<int32_t> candidate_segments = findCandidateSegments(start_key, end_key);
            vector<pair<KeyType, ValueType>> all_entries;
            bool need_retry = false;

            for (int32_t seg_idx : candidate_segments) {
                if (segments[seg_idx].is_splitting()) {
                    need_retry = true;
                    break;
                }
                auto seg_results = segments[seg_idx].rangeSearch(start_key, end_key);
                result.total_boxes_accessed += seg_results.size();
                all_entries.insert(all_entries.end(), seg_results.begin(), seg_results.end());
            }
            
            if (!need_retry) {
                result.total_keys_examined = all_entries.size();
                result.results = move(all_entries);
                return result;
            }
            
            retry_count++;
            std::this_thread::yield();
        }
        return result;
    }

    size_t scan_optimized(KeyType key_low_bound, size_t key_num, pair<KeyType, ValueType>* result) {
        if (segments.empty() || key_num == 0 || result == nullptr) {
            return 0;
        }

        const int MAX_RETRIES = 10;
        int retry_count = 0;
        
        while (retry_count < MAX_RETRIES) {
            if (is_global_splitting()) {
                std::this_thread::yield();
                retry_count++;
                continue;
            }
            int32_t start_segment_idx = searchIndex(key_low_bound);
            if (start_segment_idx < 0) {
                start_segment_idx = 0;
            }
            size_t collected = 0;
            bool need_retry = false;
            for (size_t seg_idx = start_segment_idx; seg_idx < segments.size() && collected < key_num; seg_idx++) {
                if (segments[seg_idx].is_splitting()) {
                    need_retry = true;
                    break;
                }
                size_t remaining = key_num - collected;
                bool need_filter = true;
                if (key_low_bound <= segments[seg_idx].getLowerBound()) {
                    need_filter = false;
                }
                size_t seg_collected = segments[seg_idx].scan_optimized(key_low_bound,
                                                                        remaining,
                                                                        result + collected,
                                                                        need_filter);
                collected += seg_collected;
                if (seg_collected == 0 && need_filter) {
                    if (key_low_bound > segments[seg_idx].getUpperBound()) {
                        continue;
                    } else {
                        break;
                    }
                }
            }
            if (!need_retry) {
                return collected;
            }
            retry_count++;
            std::this_thread::yield();
        }
        return 0;
    }

    size_t scan(KeyType key_low_bound, size_t key_num, pair<KeyType, ValueType>* result) {
        return scan_optimized(key_low_bound, key_num, result);
    }

    void batch_scan_optimized(const vector<pair<KeyType, int>>& scan_queries,
                              vector<vector<pair<KeyType, ValueType>>>& results) {
        if (scan_queries.empty()) return;
        vector<BatchScanQuery> batch_queries;
        batch_queries.reserve(scan_queries.size());
        for (size_t i = 0; i < scan_queries.size(); i++) {
            batch_queries.push_back(
                {scan_queries[i].first, scan_queries[i].second, static_cast<int>(i)});
        }
        sort(batch_queries.begin(),
             batch_queries.end(),
             [](const BatchScanQuery& a, const BatchScanQuery& b) {
                 return a.start_key < b.start_key;
             });
        results.resize(scan_queries.size());
        for (const auto& query : batch_queries) {
            results[query.original_index].resize(query.scan_count);
            size_t actual_count = scan_optimized(query.start_key,
                                                 query.scan_count,
                                                 results[query.original_index].data());
            results[query.original_index].resize(actual_count);
        }
    }

    void splitSegment(int32_t index, int32_t box_index) {
        auto modification_snapshot = segments[index].get_modification_snapshot();
        
        vector<pair<KeyType, ValueType>> mergedEntries = 
            segments[index].prepare_for_split_stage1(box_index);
        
        vector<Segment<KeyType, ValueType>> newSegments;
        vector<KeyType> newSegmentStartKeys;
        
        int left_count = 3;
        int right_count = 3;
        size_t numBoxes = segments[index].boxes.size();
        int merge_start = std::max(0, box_index - left_count);
        int merge_end = std::min(static_cast<int>(numBoxes) - 1, box_index + right_count);
        
        if (merge_start > 0) {
            KeyType left_lower = segments[index].getLowerBound();
            KeyType left_upper = segments[index].getBoxUpper(merge_start - 1);
            Segment<KeyType, ValueType> leftSegment(left_lower, left_upper, 
                                                    segments[index].getBoxKeyRange(), thread_num);
            leftSegment.boxes.clear();
            leftSegment.boxes.reserve(merge_start);
            for (int i = 0; i < merge_start; i++) {
                leftSegment.boxes.push_back(std::move(segments[index].boxes[i]));
            }
            newSegments.push_back(std::move(leftSegment));
            newSegmentStartKeys.push_back(left_lower);
        }
        
        KeyType merged_lower = segments[index].getBoxLower(merge_start);
        KeyType merged_upper = segments[index].getBoxUpper(merge_end);
        vector<KeyType> keys;
        keys.reserve(mergedEntries.size());
        for (const auto& entry : mergedEntries) {
            keys.push_back(entry.first);
        }
        
        Segment<KeyType, ValueType> newSegment(merged_lower, merged_upper,
                                               segments[index].getBoxKeyRange(), thread_num);
        newSegments.push_back(std::move(newSegment));
        newSegmentStartKeys.push_back(merged_lower);
        
        if (merge_end < static_cast<int>(numBoxes) - 1) {
            KeyType right_lower = segments[index].getBoxLower(merge_end + 1);
            KeyType right_upper = segments[index].getUpperBound();
            Segment<KeyType, ValueType> rightSegment(right_lower, right_upper,
                                                     segments[index].getBoxKeyRange(), thread_num);
            rightSegment.boxes.clear();
            int right_box_count = numBoxes - (merge_end + 1);
            rightSegment.boxes.reserve(right_box_count);
            for (int i = merge_end + 1; i < static_cast<int>(numBoxes); i++) {
                rightSegment.boxes.push_back(std::move(segments[index].boxes[i]));
            }
            newSegments.push_back(std::move(rightSegment));
            newSegmentStartKeys.push_back(right_lower);
        }
        
        segments.erase(segments.begin() + index);
        segments.insert(segments.begin() + index,
                       std::make_move_iterator(newSegments.begin()),
                       std::make_move_iterator(newSegments.end()));
        
        segment_start_keys.erase(segment_start_keys.begin() + index);
        segment_start_keys.insert(segment_start_keys.begin() + index,
                                 newSegmentStartKeys.begin(),
                                 newSegmentStartKeys.end());
        
        buildSearchIndex();

        for (const auto& entry : mergedEntries) {
            insertKeyValue(entry.first, entry.second);
        }
    }

    void buildSearchIndex() {
        if (segment_start_keys.empty()) return;
        
        int64_t redundantSize = segment_start_keys.size() * 90;
        vector<int32_t> temp_redundantArray(redundantSize, -1);
        
        double temp_a = static_cast<double>(redundantSize - 1) /
                    (segment_start_keys.back() - segment_start_keys.front());
        double temp_b = -temp_a * segment_start_keys.front();

        for (size_t i = 0; i < segment_start_keys.size(); i++) {
            int64_t position = static_cast<int64_t>(temp_a * segment_start_keys[i] + temp_b);
            if (position >= 0 && position < redundantSize) {
                temp_redundantArray[position] = i;
            }
        }
        
        int32_t lastValidIndex = 0;
        for (size_t i = 0; i < redundantSize; i++) {
            if (temp_redundantArray[i] == -1) {
                temp_redundantArray[i] = lastValidIndex;
            } else {
                lastValidIndex = temp_redundantArray[i];
            }
        }
        
        {
            static std::mutex index_update_mutex;
            std::lock_guard<std::mutex> lock(index_update_mutex);
            redundantArray = std::move(temp_redundantArray);
            a = temp_a;
            b = temp_b;
        }
    }

    vector<int32_t> findCandidateSegments(KeyType start_key, KeyType end_key) {
        vector<int32_t> candidates;
        int32_t start_seg = searchIndex(start_key);
        int32_t end_seg = searchIndex(end_key);
        for (int32_t i = start_seg; i <= end_seg && i < static_cast<int32_t>(segments.size()); i++) {
            KeyType seg_lower = segments[i].getLowerBound();
            KeyType seg_upper = segments[i].getUpperBound();
            if (!(seg_upper < start_key || seg_lower > end_key)) {
                candidates.push_back(i);
            }
        }
        return candidates;
    }

    int32_t searchIndex(KeyType key) {
        if (key <= segment_start_keys.front()) {
            return 0;
        } else if (key >= segment_start_keys.back()) {
            return segments.size() - 1;
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

            segments.emplace_back(lower, upper, box_range, thread_num);
            segment_start_keys.push_back(lower);
        }
        if (!segments.empty()) segment_start_keys.push_back(segments.back().getUpperBound() + 1);

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
            case InsertStatus::WRITING: return "WRITING";
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

    vector<KeyType> getSegmentStartKeys() { return segment_start_keys; }

    size_t get_index_size() const {
        size_t index_size = 0;
        index_size += redundantArray.size() * sizeof(int32_t);
        index_size += segment_start_keys.size() * sizeof(KeyType);
        for (const auto& segment : segments) {
            index_size += sizeof(KeyType) * 2;
            index_size += sizeof(size_t);
            index_size += sizeof(int);
            index_size += sizeof(std::atomic<int>) * 2;
            index_size += segment.getBoxCount() * sizeof(size_t);
            index_size += segment.getBoxCount() * sizeof(std::atomic<int>);
        }
        return index_size;
    }

    size_t get_total_size() const {
        size_t size = get_index_size();
        std::cout << "index_size: " << size << std::endl;
        size_t total_boxes_count = 0;
        size_t total_overflow_boxes_count = 0;
        for (const auto& segment : segments) {
            total_boxes_count += segment.getBoxCount();
            for (const auto& box : segment.boxes) {
                size_t overflow_count = box.getOverflowBoxCount();
                total_overflow_boxes_count += overflow_count;
                size += maxKey * sizeof(KeyType);
                size += maxKey * sizeof(uint8_t);
                size += maxKey * sizeof(ValueType);
                if (overflow_count > 0) {
                    size += overflow_count * sizeof(OverflowKeyValue<KeyType, ValueType>);
                }
            }
        }

        std::cout << "[Debug] total boxes count: " << total_boxes_count
                  << "; total overflow boxes count: " << total_overflow_boxes_count << std::endl;
        std::cout << "[Debug] Size of Box: " << sizeof(Box<KeyType, ValueType>) << std::endl;
        std::cout << "[Debug] Size of Overflow: " << sizeof(OverflowKeyValue<KeyType, ValueType>)
                  << std::endl;
        return size;
    }
};
}