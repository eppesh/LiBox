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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "segmentation.h"

#define overflowCapacity 3
#define maxKey 64

volatile int dummy;
using namespace std;

namespace liboxns {
inline void exponential_backoff(int retry_count) {
    if (retry_count > 10) retry_count = 10;
    int backoff = (1 << retry_count);
    std::this_thread::sleep_for(std::chrono::microseconds(backoff));
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
    retry_read:
        uint32_t start_version = version_lock_.load(std::memory_order_acquire);
        if (start_version & WRITE_LOCK_BIT) {
            exponential_backoff(retry_count++);
            goto retry_read;
        }
        
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
        if (start_version != version_lock_.load(std::memory_order_acquire)) {
            exponential_backoff(retry_count++);
            goto retry_read;
        }
        
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

    mutable ThreadLocalCounter operation_counter_;
    mutable std::atomic<bool> splitting_{false};
public:
    std::vector<Box<KeyType, ValueType>> boxes;

    Segment(KeyType lower, KeyType upper, size_t box_range, int thread_num)
        : lower_bound(lower), upper_bound(upper), box_key_range(box_range),
          operation_counter_(thread_num) {
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
        InsertResult ret = boxes[box_index].insertKeyValue(key, value);
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
        DeleteResult result = boxes[box_index].deleteKey(key);
        leave();
        return result;
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        if (!enter()) {
            return {SearchStatus::SPLIT, -1};
        }
        if (key < lower_bound || key > upper_bound) {
            leave();
            return {SearchStatus::OUT_OF_RANGE, -1};
        }
        
        size_t box_index = (key - lower_bound) / box_key_range;        
        SearchResult result = boxes[box_index].searchKey(key);
        leave();
        return result;
    }

    vector<pair<KeyType, ValueType>> prepare_for_split_stage1() {
        vector<pair<KeyType, ValueType>> mergedEntries;
        
        for (int i = 0; i < static_cast<int>(boxes.size()); i++) {
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
            thread_infos_[thread_id].current_epoch_.fetch_add(1, std::memory_order_acq_rel);
            thread_infos_[thread_id].doing_operation_.store(true, std::memory_order_release);
        }
        
        void leave_critical_section(int thread_id) {
            thread_infos_[thread_id].doing_operation_.store(false, std::memory_order_release);
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
                    splitSegment(seg_index);
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
            is_segment_splitting_.wait(false, std::memory_order_acquire);
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
            is_segment_splitting_.wait(false, std::memory_order_acquire);
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
            is_segment_splitting_.wait(false, std::memory_order_acquire);
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
    
    void splitSegment(int32_t seg_index) {
        auto start_time = std::chrono::high_resolution_clock::now();
        cout << "splitting segment " << seg_index << endl;
        if (!global_splitting_.exchange(true)) {
            auto* current = index_structure_.load();
            auto* segment = current->segments[seg_index];
            
            segment->wait_for_operations();
            auto mergedEntries = segment->prepare_for_split_stage1();
            if (mergedEntries.empty()) {
                unmark_segment_splitting(seg_index);
                global_splitting_.store(false);
                return;
            }

            vector<KeyType> keys;
            keys.reserve(mergedEntries.size());
            for (const auto& entry : mergedEntries) {
                keys.push_back(entry.first);
            }
            std::vector<keySegment<KeyType>> keysegments = 
                calculateSegments(keys, overflowThreshold, underflowThreshold, 15, segment->getLowerBound() , segment->getUpperBound());
            std::vector<StructSegment<KeyType>> final_segments = toStructSegment(keysegments);

            std::vector<Segment<KeyType, ValueType>*> new_segments;
            new_segments.reserve(final_segments.size());
            for (const auto& struct_seg : final_segments) {
                auto* new_seg = new Segment<KeyType, ValueType>(
                    struct_seg.seg_lower, 
                    struct_seg.seg_upper, 
                    struct_seg.box_range, 
                    thread_num
                );
                new_segments.push_back(new_seg);
            }
            
            populateSegmentsSerial(mergedEntries, new_segments);
            atomicReplaceIndexStructure(seg_index, new_segments, start_time);
            
            delete segment; 
            is_segment_splitting_.store(false, std::memory_order_release);
            is_segment_splitting_.notify_all();

            unmark_segment_splitting(seg_index);
            global_splitting_.store(false);
        }
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
};

    template <typename KeyType, typename ValueType>
    std::atomic<uint64_t> LiBox<KeyType, ValueType>::next_structure_id_{1};
}