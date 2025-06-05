#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>
#include <unordered_map>
#include <condition_variable>

#include <omp.h>
#include <xmmintrin.h>
#include <immintrin.h>

#include "libox_utils.h"

#define overflowCapacity 5
#define maxKey 64

volatile int dummy;
atomic<int> globalVersion {0};
using namespace std;

class TaskQueue {
private:
    queue<packaged_task<void()>> tasks;
    mutex mtx;
    condition_variable cv;
    bool done = false;

public:
    future<void> pushTask(function<void()> task) {
        packaged_task<void()> packagedTask(move(task));
        future<void> result = packagedTask.get_future();
        {
            unique_lock<mutex> lock(mtx);
            tasks.push(move(packagedTask));
        }
        cv.notify_one();
        return result;
    }

    bool popTask(packaged_task<void()>& task) {
        unique_lock<mutex> lock(mtx);
        cv.wait(lock, [this]{ return !tasks.empty() || done; });
        if (tasks.empty())
            return false;
        task = move(tasks.front());
        tasks.pop();
        return true;
    }

    void shutdown() {
        {
            unique_lock<mutex> lock(mtx);
            done = true;
        }
        cv.notify_all();
    }
};

auto setThreadAffinity(thread &thr, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(thr.native_handle(), sizeof(cpu_set_t), &cpuset);
}

namespace liboxns {
enum class InsertStatus {
    SUCCESS,
    FULL,
    BOUND_ERROR,
    SPLIT,
};

struct InsertResult {
    InsertStatus status; 
    int box_index;       
};

enum class SearchStatus {
    SUCCESS,
    NOT_FOUND, 
    ERROR   
};

template <typename KeyType, typename ValueType>
struct SearchResult {
    SearchStatus status; 
    ValueType value;        
};

class NoOpLock {
public:
    void lock() { }
    void unlock() { }
};

template <typename KeyType, typename ValueType>
class OverflowKeyValue {
private:
    size_t currentSize = 0;
    atomic<int> overVersion{0};
    alignas(64) array<uint8_t, maxKey> keys_low;

    size_t avx512_filter_keys_optimized(KeyType key_low_bound, 
                                     pair<KeyType, ValueType>* result_buffer,
                                     size_t max_results) const {
        _mm_prefetch((const char*)&keys[0], _MM_HINT_T0);
        _mm_prefetch((const char*)&values[0], _MM_HINT_T0);
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
        size_t to_copy = min(currentSize, max_count);
        
        for (size_t i = 0; i < to_copy; i++) {
            result[i] = {keys[i], values[i]};
        }
        
        return to_copy;
    }

public:
    alignas(64) array<ValueType, maxKey> values;
    alignas(64) array<KeyType, maxKey> keys;
    size_t getTotalCount() const {
        return currentSize;
    }

    OverflowKeyValue() {
    }

    size_t copyAllData(pair<KeyType, ValueType>* result) const {
        size_t i = 0;
        for (; i + 8 <= currentSize; i += 8) {
            __m512i keys_vec = _mm512_load_si512(&keys[i]);
            __m512i values_vec = _mm512_load_si512(&values[i]);
            for (int j = 0; j < 8; j++) {
                result[i + j] = {keys[i + j], values[i + j]};
            }
        }
        for (; i < currentSize; i++) {
            result[i] = {keys[i], values[i]};
        }
        return currentSize;
    }

    int findKeyIndex(KeyType key) const {
        uint8_t target_low = static_cast<uint8_t>(key & 0xFF);
        
        __m512i v_target_low = _mm512_set1_epi8(target_low);
        __m512i v_keys_low = _mm512_load_si512(reinterpret_cast<const __m512i*>(keys_low.data()));
        __mmask64 mask_low = _mm512_cmpeq_epi8_mask(v_keys_low, v_target_low);

        while (mask_low) {
            int candidate = __builtin_ctzll(mask_low);
            mask_low &= mask_low - 1;
            if (keys[candidate] == key)
                return (candidate < currentSize) ? candidate : -1;
        }
        return -1;
    }

    void updateValueAt(int index, ValueType value) {
        while (overVersion.load(memory_order_acquire) == 1) {}
        overVersion.store(1, memory_order_release);
        values[index] = value;
        overVersion.store(0, memory_order_release);
    }

    size_t scan_optimized(KeyType key_low_bound, size_t max_count, 
                         pair<KeyType, ValueType>* result,
                         bool need_filter = true) const {
        if (max_count == 0 || result == nullptr || currentSize == 0) {
            return 0;
        }
        
        if (!need_filter) {
            return copyDataWithLimit(result, max_count);
        } else {
            return avx512_filter_keys_optimized(key_low_bound, result, max_count);
        }
    }

    size_t scan(KeyType key_low_bound, size_t max_count, 
                pair<KeyType, ValueType>* result, 
                bool need_filter = true) const {
        return scan_optimized(key_low_bound, max_count, result, need_filter);
    }

    size_t scan_load(pair<KeyType, ValueType>* result, size_t max_count) const {
        return copyDataWithLimit(result, max_count);
    }

    InsertResult insert(KeyType key, ValueType value, atomic<int>* vecPtr) {
        while (overVersion.load(memory_order_acquire) == 1) {}
        if (currentSize < maxKey) {
            overVersion.store(1, memory_order_release);
            if (vecPtr->load(memory_order_acquire) == 1 or globalVersion.load(memory_order_acquire) == 1) {
                overVersion.store(0, memory_order_release);
                return {InsertStatus::SPLIT, 1};
            }
            keys[currentSize] = key;
            keys_low[currentSize] = static_cast<uint8_t>(key & 0xFF);         // lower 8 bits
            values[currentSize] = value;
            currentSize++;
            overVersion.store(0, memory_order_release);
            return {InsertStatus::SUCCESS, 1};
        }

        return {InsertStatus::FULL, -1};
    }

    vector<pair<KeyType, ValueType>> rangeSearch(KeyType start_key, KeyType end_key) const {
        vector<pair<KeyType, ValueType>> results;
        
        const size_t batch_size = 8;
        size_t num_batches = (currentSize + batch_size - 1) / batch_size;
        
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
                
                if (actual_idx < currentSize) {
                    results.push_back({keys[actual_idx], values[actual_idx]});
                }
                
                mask_in_range &= mask_in_range - 1;
            }
        }
        
        return results;
    }

    SearchResult<KeyType, ValueType> search(KeyType key) const {
        int index = findKeyIndex(key);
        if (index >= 0) {
            return {SearchStatus::SUCCESS, values[index]};
        }
        return {SearchStatus::NOT_FOUND, std::numeric_limits<ValueType>::max()};
    }

    size_t size() const { return currentSize; }

    vector<pair<KeyType, ValueType>> getEntries() const {
        vector<pair<KeyType, ValueType>> entries;
        for (size_t i = 0; i < currentSize; i++) {
            entries.push_back({ keys[i], values[i] });
        }
        return entries;
    }

    std::unique_ptr<OverflowKeyValue<KeyType, ValueType>> clone() const {
        auto newObj = std::make_unique<OverflowKeyValue<KeyType, ValueType>>();
        newObj->currentSize = currentSize;
        for (size_t i = 0; i < currentSize; i++) {
            newObj->keys[i] = keys[i];
            newObj->keys_low[i] = keys_low[i];
            newObj->values[i] = values[i];
        }
        return newObj;
    }
};

template <typename KeyType, typename ValueType>
class InnerOverflowArray {
private:
    size_t capacity;
    thread_local static unique_ptr<pair<KeyType, ValueType>[]> temp_buffer;
    thread_local static size_t temp_buffer_size;
    
    void ensure_temp_buffer(size_t required_size) {
        if (!temp_buffer || temp_buffer_size < required_size) {
            temp_buffer_size = max(required_size, size_t(maxKey * overflowCapacity * 2));
            temp_buffer = make_unique<pair<KeyType, ValueType>[]>(temp_buffer_size);
        }
    }
public:
    array<unique_ptr<OverflowKeyValue<KeyType, ValueType>>, overflowCapacity> data;

    struct KeyLocation {
        int overflowIndex;  
        int keyIndex;       
        bool found;       
    };

    InnerOverflowArray() : capacity(1) { 
        data[0] = make_unique<OverflowKeyValue<KeyType, ValueType>>();
    }

    InnerOverflowArray(size_t initCapacity)
        : capacity(initCapacity)
    {
        if (capacity == 0) {
            capacity = 1;
        }
        for (size_t i = 0; i < capacity; i++) {
            data[i] = make_unique<OverflowKeyValue<KeyType, ValueType>>();
        }
    }
    
    InnerOverflowArray(const InnerOverflowArray &other) = delete;
    InnerOverflowArray& operator=(const InnerOverflowArray &other) = delete;

    InnerOverflowArray(InnerOverflowArray &&other) noexcept
        : capacity(other.capacity)
    {
        for (size_t i = 0; i < capacity; i++) {
            data[i] = move(other.data[i]);
        }
        other.capacity = 0;
    }
    
    InnerOverflowArray& operator=(InnerOverflowArray &&other) noexcept {
        if (this != &other) {
            capacity = other.capacity;
            for (size_t i = 0; i < capacity; i++) {
                data[i] = move(other.data[i]);
            }
            other.capacity = 0;
        }
        return *this;
    }

    unique_ptr<InnerOverflowArray<KeyType, ValueType>> clone() const {
        auto newArray = std::make_unique<InnerOverflowArray<KeyType, ValueType>>(capacity);
        for (size_t i = 0; i < capacity; i++) {
            newArray->data[i] = data[i]->clone();
        }
        return newArray;
    }

    KeyLocation findKeyLocation(KeyType key) const {
        for (size_t i = 0; i < capacity; i++) {
            int keyIndex = data[i]->findKeyIndex(key);
            if (keyIndex >= 0) {                                                   
                return {static_cast<int>(i), keyIndex, true};
            }
        }
        return {-1, -1, false};
    }

    bool updateValue(KeyType key, ValueType value) {
        KeyLocation location = findKeyLocation(key);
        if (location.found) {
            data[location.overflowIndex]->updateValueAt(location.keyIndex, value);
            return true;
        }
        return false;
    }

    size_t getTotalCount() const {
        size_t total = 0;
        for (size_t i = 0; i < capacity; i++) {
            total += data[i]->getTotalCount();
        }
        return total;
    }

    size_t scan_optimized(KeyType key_low_bound, size_t max_count, 
                         pair<KeyType, ValueType>* result,
                         bool need_filter = true) {
        if (max_count == 0 || result == nullptr) {
            return 0;
        }
        
        size_t collected = 0;
        for (size_t i = 0; i < capacity && collected < max_count; i++) {
            size_t box_collected = data[i]->scan_optimized(
                key_low_bound, 
                max_count - collected, 
                result + collected,
                true  // need_filter = true
            );
            collected += box_collected;
        }
        return collected;
        
    }

    size_t scan(KeyType key_low_bound, size_t max_count, 
                pair<KeyType, ValueType>* result, 
                bool need_filter = true) {
        return scan_optimized(key_low_bound, max_count, result, need_filter);
    }

    InsertResult insert(KeyType key, ValueType value, atomic<int>* vecPtr) {
        if (this == nullptr || reinterpret_cast<uintptr_t>(this) < 0x1000) {
            cerr << "Critical error: Invalid InnerOverflowArray this pointer: " 
                 << this << " in thread " << this_thread::get_id() << endl;
            return {InsertStatus::BOUND_ERROR, -1};
        }

        try {
            if (capacity == 0 || !data[0]) {
                cerr << "Warning: InnerOverflowArray capacity is 0 or data[0] is null" << endl;
                capacity = 1;
                data[0] = make_unique<OverflowKeyValue<KeyType, ValueType>>();
            }

            bool updated = updateValue(key, value);
            if (updated) {
                return {InsertStatus::SUCCESS, 1};
            }
            InsertResult ret;
            size_t box_size = data[capacity-1]->size();
            if (box_size < maxKey) {
                return data[capacity - 1]->insert(key, value, vecPtr);
            } else {
                if (capacity >= overflowCapacity) {
                    return {InsertStatus::FULL, -1};
                } else {
                    data[capacity] = make_unique<OverflowKeyValue<KeyType, ValueType>>();
                    InsertResult ret = data[capacity]->insert(key, value, vecPtr);
                    capacity++;
                    return ret;
                }
            }
        } catch (const exception& e) {
            cerr << "Exception in InnerOverflowArray::insert: " << e.what() << endl;
            return {InsertStatus::BOUND_ERROR, -1};
        } catch (...) {
            cerr << "Unknown exception in InnerOverflowArray::insert" << endl;
            return {InsertStatus::BOUND_ERROR, -1};
        }
    }

    vector<pair<KeyType, ValueType>> rangeSearch(KeyType start_key, KeyType end_key) const {
        vector<pair<KeyType, ValueType>> results;
        
        for (size_t i = 0; i < capacity; i++) {
            auto box_results = data[i]->rangeSearch(start_key, end_key);
            results.insert(results.end(), box_results.begin(), box_results.end());
        }
        
        return results;
    }

    SearchResult<KeyType, ValueType> search(KeyType key) {
        for (size_t i = 0; i < capacity; i++) {
            SearchResult ret = data[i]->search(key);
            if (ret.status == SearchStatus::SUCCESS)
                return ret;
        }
        return {SearchStatus::NOT_FOUND, std::numeric_limits<ValueType>::max()};
    }

    size_t size () const {
        if (capacity == 1 && data[0].get ()->size () == 0) {
            return 0;
        }
        return capacity;
    }

    OverflowKeyValue<KeyType, ValueType>* get(size_t index) {
        return data[index].get();
    }

    vector<pair<KeyType, ValueType>> getEntries() const {
        vector<pair<KeyType, ValueType>> entries;
        for (int i = 0; i < capacity; i++){
            vector<pair<KeyType, ValueType>> be = data[i]->getEntries();
            entries.insert(entries.end(), be.begin(), be.end());
        }
        return entries;
    }

    size_t scan_load(pair<KeyType, ValueType>* result, size_t max_count) const {
        size_t collected = 0;
        for (size_t i = 0; i < capacity && collected < max_count; i++) {
            size_t box_collected = data[i]->scan_load(
                result + collected, 
                max_count - collected
            );
            collected += box_collected;
        }
        return collected;
    }
};

template <typename KeyType, typename ValueType>
thread_local unique_ptr<pair<KeyType, ValueType>[]> InnerOverflowArray<KeyType, ValueType>::temp_buffer;

template <typename KeyType, typename ValueType>
thread_local size_t InnerOverflowArray<KeyType, ValueType>::temp_buffer_size = 0;

template <typename KeyType, typename ValueType>
class Box {
private:  
    size_t currentSize = 0;
    atomic<int> boxVersion{0};

    alignas(64) array<KeyType, maxKey> keys; 
    alignas(64) array<uint8_t, maxKey> keys_low;
    alignas(64) array<ValueType, maxKey> values;    

    unique_ptr<InnerOverflowArray<KeyType, ValueType>> sharedOverflow;

    thread_local static unique_ptr<pair<KeyType, ValueType>[]> box_temp_buffer;
    thread_local static size_t box_temp_buffer_size;

    void ensure_box_temp_buffer(size_t required_size) const {
        if (!box_temp_buffer || box_temp_buffer_size < required_size) {
            box_temp_buffer_size = max(required_size, size_t(maxKey * 3));
            box_temp_buffer = make_unique<pair<KeyType, ValueType>[]>(box_temp_buffer_size);
        }
    }

    int findKeyIndex(KeyType key) const {
        uint8_t target_low = static_cast<uint8_t>(key & 0xFF);

        __m512i v_target_low = _mm512_set1_epi8(target_low);
        __m512i v_keys_low = _mm512_load_si512(reinterpret_cast<const __m512i*>(keys_low.data()));
        __mmask64 mask_low = _mm512_cmpeq_epi8_mask(v_keys_low, v_target_low);

        while (mask_low) {
            int candidate = __builtin_ctzll(mask_low);
            mask_low &= mask_low - 1;
            if (keys[candidate] == key)
                return (candidate < currentSize) ? candidate : -1;
        }
        return -1;
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
        for (size_t i = 0; i < currentSize; i++) {
            result[i] = {keys[i], values[i]};
        }
        return currentSize;
    }

public: 
    Box() : sharedOverflow(make_unique<InnerOverflowArray<KeyType, ValueType>>()) { }

    Box(const Box &other)
        : values(other.values),
          currentSize(other.currentSize),
          keys(other.keys),
          keys_low(other.keys_low)
    {
        if (other.sharedOverflow) {
            sharedOverflow = other.sharedOverflow->clone();
        } else {
            sharedOverflow = make_unique<InnerOverflowArray<KeyType, ValueType>>();
        }
    }

    Box& operator=(const Box &other) {
        if (this != &other) {
            values = other.values;
            currentSize = other.currentSize;
            keys = other.keys;
            keys_low = other.keys_low;
            if (other.sharedOverflow) {
                sharedOverflow = other.sharedOverflow->clone();
            } else {
                sharedOverflow.reset();
            }
        }
        return *this;
    }

    Box(Box &&other) noexcept = default;
    Box& operator=(Box &&other) noexcept = default;
    ~Box() = default;

    size_t getTotalCount() const {
        size_t total = currentSize;
        if (sharedOverflow) {
            total += sharedOverflow->getTotalCount();
        }
        return total;
    }

    size_t scan_optimized(KeyType key_low_bound, size_t max_count, 
                         pair<KeyType, ValueType>* result) const {
        if (max_count == 0 || result == nullptr) {
            return 0;
        }

        size_t collected = 0;
        collected = avx512_filter_main_keys_optimized(key_low_bound, result, max_count);
        
        if (sharedOverflow && collected < max_count) {
            collected += sharedOverflow->scan_optimized(
                key_low_bound, 
                max_count - collected, 
                result + collected,
                true
            );
        }   
        return collected;
    }

    size_t scan(KeyType key_low_bound, size_t max_count, 
                pair<KeyType, ValueType>* result, 
                bool need_filter = true) const {
        return scan_optimized(key_low_bound, max_count, result, need_filter);
    }

    InsertResult insertKeyValue(KeyType key, ValueType value, atomic<int>* vecPtr) {
        while (boxVersion.load(memory_order_acquire) == 1) {}
        boxVersion.store(1, memory_order_release);

        try {
            int existingIndex = findKeyIndex(key);
            if (existingIndex >= 0) {
                if (vecPtr->load(memory_order_acquire) == 1 || globalVersion.load(memory_order_acquire) == 1) {
                    boxVersion.store(0, memory_order_release);
                    return {InsertStatus::SPLIT, 1};
                }
                values[existingIndex] = value;
                boxVersion.store(0, memory_order_release);
                return {InsertStatus::SUCCESS, 1};
            }

            if (currentSize < maxKey) {
                if (vecPtr->load(memory_order_acquire) == 1 || globalVersion.load(memory_order_acquire) == 1) {
                    boxVersion.store(0, memory_order_release);
                    return {InsertStatus::SPLIT, 1};
                }
                keys[currentSize] = key;
                keys_low[currentSize] = static_cast<uint8_t>(key & 0xFF);
                values[currentSize] = value;
                currentSize++;
                boxVersion.store(0, memory_order_release);
                return {InsertStatus::SUCCESS, 1};
            }

            if (!sharedOverflow) {
                sharedOverflow = make_unique<InnerOverflowArray<KeyType, ValueType>>();
            }
            if (!sharedOverflow || reinterpret_cast<uintptr_t>(sharedOverflow.get()) < 0x1000) {
                cerr << "Warning: Invalid sharedOverflow pointer detected in thread "
                     << this_thread::get_id() << endl;
                
                sharedOverflow = make_unique<InnerOverflowArray<KeyType, ValueType>>();
            }
            
            InsertResult ret = sharedOverflow->insert(key, value, vecPtr);
            boxVersion.store(0, memory_order_release);
            return ret;
        } catch (const exception& e) {
            cerr << "Exception in insertKeyValue: " << e.what() << endl;
            boxVersion.store(0, memory_order_release);
            return {InsertStatus::BOUND_ERROR, -1};
        }catch (...) {
            cerr << "Unknown exception in insertKeyValue" << endl;
            boxVersion.store(0, memory_order_release);
            return {InsertStatus::BOUND_ERROR, -1};
        }
    }

    vector<pair<KeyType, ValueType>> rangeSearch(KeyType start_key, KeyType end_key) const {
        vector<pair<KeyType, ValueType>> results;
        
        const size_t batch_size = 8;
        size_t num_batches = (currentSize + batch_size - 1) / batch_size;
        
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
                
                if (actual_idx < currentSize) {
                    results.push_back({keys[actual_idx], values[actual_idx]});
                }
                
                mask_in_range &= mask_in_range - 1;
            }
        }
        
        if (sharedOverflow) {
            auto overflow_results = sharedOverflow->rangeSearch(start_key, end_key);
            results.insert(results.end(), overflow_results.begin(), overflow_results.end());
        }
        
        return results;
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) const {
        int index = findKeyIndex(key);
        if (index >= 0) {
            return {SearchStatus::SUCCESS, values[index]};
        }

        if (sharedOverflow) {
            return sharedOverflow->search(key);
        }
        
        return {SearchStatus::NOT_FOUND, std::numeric_limits<ValueType>::max()};
    }

    size_t getCurrentSize() const { return currentSize; }
 
    vector<pair<KeyType, ValueType>> getEntries() const {
        vector<pair<KeyType, ValueType>> entries;
        for (size_t i = 0; i < currentSize; i++) {
            entries.push_back({ keys[i], values[i] });
        }
        vector<pair<KeyType, ValueType>> entry = sharedOverflow->getEntries();
        entries.insert(entries.end(), entry.begin(), entry.end());
        return entries;
    }

    size_t scan_load(pair<KeyType, ValueType>* result, size_t max_count) const {
        size_t collected = copyMainKeysWithLimit(result, max_count);
        
        if (sharedOverflow) {
            collected += sharedOverflow->scan_load(
                result + collected, 
                max_count - collected
            );
        }
        
        return collected;
    }

    size_t getOverflowBoxCount () const { return sharedOverflow->size (); }
};

template <typename KeyType, typename ValueType>
thread_local unique_ptr<pair<KeyType, ValueType>[]> Box<KeyType, ValueType>::box_temp_buffer;

template <typename KeyType, typename ValueType>
thread_local size_t Box<KeyType, ValueType>::box_temp_buffer_size = 0;

template <typename KeyType, typename ValueType>
class Segment {
private:
    KeyType lower_bound;
    KeyType upper_bound;
    size_t box_key_range;
    int numBoxes;

public:
    std::vector<Box<KeyType, ValueType>> boxes;
    atomic<int> version {0};
    atomic<int> versionForBox {0};

    Segment(KeyType lower, KeyType upper, size_t box_range)
        : lower_bound(lower), upper_bound(upper), box_key_range(box_range) {
        size_t total = upper - lower + 1;
        size_t box_count = total / box_range;
        if (total % box_range != 0)
            box_count++;        
        numBoxes = box_count;
        boxes.resize(box_count);
    }

    Segment(const Segment &) = delete;
    Segment& operator=(const Segment &) = delete;

    Segment(Segment&& other) noexcept
        : lower_bound(other.lower_bound),
          upper_bound(other.upper_bound),
          box_key_range(other.box_key_range),
          numBoxes(other.numBoxes),
          boxes(std::move(other.boxes)),
          version(other.version.load()),
          versionForBox(other.versionForBox.load())
    {
    }

    Segment& operator=(Segment&& other) noexcept {
        if (this != &other) {
            lower_bound = other.lower_bound;
            upper_bound = other.upper_bound;
            box_key_range = other.box_key_range;
            numBoxes = other.numBoxes;
            boxes = std::move(other.boxes);
            version.store(other.version.load());
            versionForBox.store(other.versionForBox.load());
        }
        return *this;
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) {
        while (version.load(memory_order_acquire) == 1) {}
        size_t box_index = (key - lower_bound) / box_key_range;
        InsertResult ret = boxes[box_index].insertKeyValue(key, value, &versionForBox);
        if (ret.status == InsertStatus::FULL) {
            ret.box_index = box_index;
        }
        return ret;
    }

    vector<pair<KeyType, ValueType>> rangeSearch(KeyType start_key, KeyType end_key) const {
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

    size_t scan_optimized(KeyType key_low_bound, size_t max_count, 
                         pair<KeyType, ValueType>* result,
                         bool need_filter = true) const {
        if (max_count == 0 || result == nullptr || boxes.empty()) {
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
        
        for (size_t box_idx = start_box_idx; 
             box_idx < boxes.size() && collected < max_count; 
             box_idx++) {
            
            size_t remaining = max_count - collected;
            
            bool box_need_filter = need_filter;
            
            KeyType box_lower = getBoxLower(box_idx);
            if (key_low_bound <= box_lower) {
                box_need_filter = false;
            }
            
            if (!box_need_filter) {
                size_t box_total = boxes[box_idx].getTotalCount();
                
                if (box_total <= remaining) {
                    size_t box_collected = boxes[box_idx].scan_load(
                        result + collected,
                        remaining
                    );
                    collected += box_collected;
                } else {
                    size_t box_collected = boxes[box_idx].scan_optimized(
                        key_low_bound, 
                        remaining, 
                        result + collected
                    );
                    collected += box_collected;
                    break;
                }
            } else {
                size_t box_collected = boxes[box_idx].scan_optimized(
                    key_low_bound, 
                    remaining, 
                    result + collected
                );
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

    size_t scan(KeyType key_low_bound, size_t max_count, 
                pair<KeyType, ValueType>* result,
                bool need_filter = true) const {
        return scan_optimized(key_low_bound, max_count, result, need_filter);
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) const {
        size_t box_index = (key - lower_bound) / box_key_range;
        return boxes[box_index].searchKey(key);
    }

    vector<pair<KeyType, ValueType>> getAllEntries() const {
        vector<pair<KeyType, ValueType>> entries;
        for (const auto &box : boxes) {
            vector<pair<KeyType, ValueType>> be = box.getEntries();
            entries.insert(entries.end(), be.begin(), be.end());
        }
        return entries;
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

    TaskQueue *splitTaskQueue;
    int thread_num;
    
    vector<future<void>> currentBatchTasks;

    struct BatchScanQuery {
        KeyType start_key;
        int scan_count;
        int original_index;
    };
public:
    struct RangeSearchResult {
        vector<pair<KeyType, ValueType>> results;
        size_t total_boxes_accessed;
        size_t total_keys_examined;
        
        RangeSearchResult() : total_boxes_accessed(0), total_keys_examined(0) {}
    };

    LiBox() {}

    LiBox(int uThreshold, int oThreshold, TaskQueue *splitTaskQueue, int thread_num)
        : underflowThreshold(uThreshold), overflowThreshold(oThreshold), splitTaskQueue(splitTaskQueue),
        thread_num(thread_num) { }

    ~LiBox() {
        segments.clear();
        segment_start_keys.clear();
    }

    void init(int uThreshold, int oThreshold, TaskQueue *taskQueue, int threadNum){
        underflowThreshold = uThreshold;
        overflowThreshold = oThreshold;
        splitTaskQueue = taskQueue;
        thread_num = threadNum;
    }

    void buildSearchIndex() {
        int64_t redundantSize = segment_start_keys.size() * 90;
        redundantArray.resize(redundantSize, -1);
        
        a = static_cast<double>(redundantSize - 1) / (segment_start_keys.back() - segment_start_keys.front());
        b = -a * segment_start_keys.front();
        
        for (size_t i = 0; i < segment_start_keys.size(); i++) {
            int64_t position = static_cast<int64_t>(a * (segment_start_keys)[i] + b);
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

    RangeSearchResult rangeSearch(KeyType start_key, KeyType end_key) {
        RangeSearchResult result;
        
        if (start_key > end_key) {
            return result;
        }
        vector<int32_t> candidate_segments = findCandidateSegments(start_key, end_key);
        vector<pair<KeyType, ValueType>> all_entries;
        
        for (int32_t seg_idx : candidate_segments) {
            auto seg_results = segments[seg_idx].rangeSearch(start_key, end_key);
            result.total_boxes_accessed += seg_results.size();
            all_entries.insert(all_entries.end(), seg_results.begin(), seg_results.end());
        }
        result.total_keys_examined = all_entries.size();
        result.results = move(all_entries);        
        return result;
    }

    size_t scan_optimized(KeyType key_low_bound, size_t key_num, 
                         pair<KeyType, ValueType>* result) {
        if (segments.empty() || key_num == 0 || result == nullptr) {
            return 0;
        }
        
        int32_t start_segment_idx = searchIndex(key_low_bound);
        if (start_segment_idx < 0) {
            start_segment_idx = 0;
        }
        
        size_t collected = 0;
        
        for (size_t seg_idx = start_segment_idx; 
             seg_idx < segments.size() && collected < key_num; 
             seg_idx++) {
            
            size_t remaining = key_num - collected;
            
            bool need_filter = true;
            
            if (key_low_bound <= segments[seg_idx].getLowerBound()) {
                need_filter = false;
            }
            
            size_t seg_collected = segments[seg_idx].scan_optimized(
                key_low_bound, 
                remaining, 
                result + collected,
                need_filter
            );
            
            collected += seg_collected;
            
            if (seg_collected == 0 && need_filter) {
                if (key_low_bound > segments[seg_idx].getUpperBound()) {
                    continue;
                } else {
                    break;
                }
            }
        }
        
        return collected;
    }

    size_t scan(KeyType key_low_bound, size_t key_num, 
                pair<KeyType, ValueType>* result) {
        return scan_optimized(key_low_bound, key_num, result);
    }

    void batch_scan_optimized(const vector<pair<KeyType, int>>& scan_queries,
                              vector<vector<pair<KeyType, ValueType>>>& results) {
        if (scan_queries.empty()) return;
        
        vector<BatchScanQuery> batch_queries;
        batch_queries.reserve(scan_queries.size());
        
        for (size_t i = 0; i < scan_queries.size(); i++) {
            batch_queries.push_back({scan_queries[i].first, scan_queries[i].second, static_cast<int>(i)});
        }
        
        sort(batch_queries.begin(), batch_queries.end(),
             [](const BatchScanQuery& a, const BatchScanQuery& b) {
                 return a.start_key < b.start_key;
             });
        
        results.resize(scan_queries.size());
        
        for (const auto& query : batch_queries) {
            results[query.original_index].resize(query.scan_count);
            size_t actual_count = scan_optimized(query.start_key, query.scan_count, 
                                               results[query.original_index].data());
            results[query.original_index].resize(actual_count);
        }
    }

    int32_t searchIndex(KeyType key) {
        if(key <= segment_start_keys.front()){
            return 0;
        } else if(key >= segment_start_keys.back()){
            return segments.size()-1; // return the last segment;
        }
        int64_t position = static_cast<int64_t>(a * key + b);
        int32_t estimatedIndex = redundantArray[position];

        if (segment_start_keys[estimatedIndex] <= key && segment_start_keys[estimatedIndex+1] > key) {
            return estimatedIndex;
        }

        if (segment_start_keys[estimatedIndex-1] <= key && segment_start_keys[estimatedIndex] > key) {
            return estimatedIndex - 1;
        }

        if (segment_start_keys[estimatedIndex+1] <= key && segment_start_keys[estimatedIndex+2] > key) {
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
                if (segment_start_keys[mid] <= key && (mid + 1 >= segment_start_keys.size() || 
                                                     segment_start_keys[mid + 1] > key)) {
                    return mid; 
                }
                
                if (segment_start_keys[mid] <= key) {
                    low = mid + 1; 
                } else {
                    high = mid - 1; 
                }
            }
        } 
        else {
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
                if (segment_start_keys[mid] <= key && (mid + 1 >= segment_start_keys.size() || 
                                                     segment_start_keys[mid + 1] > key)) {
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

    void loadConfigByFile(const string &config_file) {
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
                if constexpr (std::is_same_v<KeyType, double>){
                    lower = std::stod(token);
                } else if constexpr (std::is_signed_v<KeyType>){
                    lower = std::stoll(token);
                } else {
                    lower = std::stoull(token);
                }
            }
            if (getline(iss, token, ',')) {
                if constexpr (std::is_same_v<KeyType, double>){
                    upper = std::stod(token);
                } else if constexpr (std::is_signed_v<KeyType>){
                    upper = std::stoll(token);
                } else {
                    upper = std::stoull(token);
                }
            }
            if (getline(iss, token)) {
                box_range = stoul(token);
            }

            segments.emplace_back(lower, upper, box_range);
            segment_start_keys.push_back(lower);
        }
        if (!segments.empty())
            segment_start_keys.push_back(segments.back().getUpperBound() + 1);  

        buildSearchIndex();
    }

    void bulk_load(std::pair<KeyType, ValueType> *key_value, size_t num){
        size_t inserted = 0;
        omp_set_num_threads(thread_num);
        #pragma omp parallel for reduction(+:inserted)
        for(int i=0; i<num; ++i){
            KeyType key = key_value[i].first;
            ValueType value = key_value[i].second;
            if(insertKeyValue(key, value).status == InsertStatus::SUCCESS){
                inserted++;
            }
        }
        std::cout << "bulk loading finished! total num: " << num << ", inserted " << inserted << " keys \n";
    }

    void buildIndex(vector<KeyType> *file_keys) {
        int keys_size = file_keys->size();
        int inserted = 0;
        omp_set_num_threads(thread_num);
        #pragma omp parallel for reduction(+:inserted)
        for (int i = 0; i < keys_size; i++) {
            if (insertKeyValue((*file_keys)[i], 1).status == InsertStatus::SUCCESS)
                inserted++;
        }
        cout << "bulk loading finished, inserted " << inserted << " keys \n";
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) {
        while (globalVersion.load(memory_order_acquire) == 1) {}
        int32_t num_index = searchIndex(key);
        InsertResult ret = segments[num_index].insertKeyValue(key, value);
        if (ret.status == InsertStatus::FULL) {
            int32_t box_index = ret.box_index;
            future<void> task_future = splitTaskQueue->pushTask([this, num_index, box_index, key, value]() {
                this->splitSegment(num_index, box_index);
                this->insertKeyValue(key, value);
            });
            
            currentBatchTasks.push_back(move(task_future));
            ret.status = InsertStatus::SUCCESS;
        }

        if (ret.status == InsertStatus::SPLIT) {
            future<void> task_future = splitTaskQueue->pushTask([this, key, value]() {
                this->insertKeyValue(key, value);
            });
            currentBatchTasks.push_back(move(task_future));
            ret.status = InsertStatus::SUCCESS;
        }
        return ret;
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        int32_t num_index = searchIndex(key);
        return segments[num_index].searchKey(key);
    }

    void splitSegment(int32_t index, int32_t box_index) {
        vector<Segment<KeyType, ValueType>> newSegments;
        vector<KeyType> newsegment_start_keys;
        vector<pair<KeyType, ValueType>> mergedEntries;

        //The first stage is dealing with the segment
        {
            segments[index].version.store(1, memory_order_release);
            segments[index].versionForBox.store(1, memory_order_release);
            // Custom number
            int left_count = 3;
            int right_count = 3;

            size_t numBoxes = segments[index].boxes.size();
            int merge_start = (box_index - left_count >= 0) ? box_index - left_count : 0;
            int merge_end = (box_index + right_count < static_cast<int>(numBoxes)) ? box_index + right_count : numBoxes - 1;
            if (merge_start > 0) {
                KeyType left_lower = segments[index].getLowerBound();
                KeyType left_upper = segments[index].getBoxUpper(merge_start - 1);
                Segment<KeyType, ValueType> leftSegment(left_lower, left_upper, segments[index].getBoxKeyRange());
                leftSegment.boxes.clear();
                for (int i = 0; i < merge_start; i++) {
                    leftSegment.boxes.push_back(segments[index].boxes[i]);
                }
                newSegments.push_back(move(leftSegment));
                newsegment_start_keys.push_back(leftSegment.getLowerBound());
            }
            KeyType merged_lower = segments[index].getBoxLower(merge_start);
            KeyType merged_upper = segments[index].getBoxUpper(merge_end);
            Segment<KeyType, ValueType> mergedSegment(merged_lower, merged_upper, segments[index].getBoxKeyRange());
            mergedSegment.boxes.clear();

            for (int i = merge_start; i <= merge_end; i++) {
                auto entries = segments[index].boxes[i].getEntries();
                mergedEntries.insert(mergedEntries.end(), entries.begin(), entries.end());
            }

            sort(mergedEntries.begin(), mergedEntries.end(),
                [](const pair<KeyType, ValueType>& a, const pair<KeyType, ValueType>& b) {
                    return a.first < b.first;
                });

            vector<KeyType> keys;
            keys.reserve(mergedEntries.size());
            for (const auto &p : mergedEntries) {
                keys.push_back(p.first);
            }
            int maxMergeCount = 3;
            double underflowThreshold = 1;
            double overflowThreshold = 0.05;
            vector<Block<KeyType>> blocks = computeBlocks(keys, 32, merged_lower, merged_upper);
            vector<StructSegment<KeyType>> initSegments = partitionSegmentsOverall(blocks, keys, underflowThreshold, maxMergeCount);
            vector<StructSegment<KeyType>> finalSegments = expandSegments(initSegments, blocks, keys, underflowThreshold, overflowThreshold, maxMergeCount);
            for (int i = 0; i < finalSegments.size(); i++) {
                Segment<KeyType, ValueType> newSegment(finalSegments[i].seg_lower, finalSegments[i].seg_upper, finalSegments[i].box_range);
                newSegments.push_back(move(newSegment));
                newsegment_start_keys.push_back(newSegment.getLowerBound());
            }

            if (merge_end < static_cast<int>(numBoxes) - 1) {
                KeyType right_lower = segments[index].getBoxLower(merge_end + 1);
                KeyType right_upper = segments[index].getUpperBound();
                Segment<KeyType, ValueType> rightSegment(right_lower, right_upper, segments[index].getBoxKeyRange());
                rightSegment.boxes.clear();
                for (int i = merge_end + 1; i < static_cast<int>(numBoxes); i++) {
                    rightSegment.boxes.push_back(segments[index].boxes[i]);
                }
                newSegments.push_back(move(rightSegment));
                newsegment_start_keys.push_back(rightSegment.getLowerBound());
            }

            segments[index].versionForBox.store(0, memory_order_release);
            segments[index].version.store(0, memory_order_release);
        }

        // The second stage, global update
        {   
            globalVersion.store(1, memory_order_release);
            segments.erase(segments.begin() + index);
            segments.insert(segments.begin() + index,
                  std::make_move_iterator(newSegments.begin()),
                  std::make_move_iterator(newSegments.end()));            
            segment_start_keys.erase(segment_start_keys.begin() + index);
            segment_start_keys.insert(segment_start_keys.begin() + index, newsegment_start_keys.begin(), newsegment_start_keys.end());
            buildSearchIndex();
            globalVersion.store(0, memory_order_release);
        }

        // The third stage is to reinsert the data
        {

            for (int i = 0; i < mergedEntries.size(); i ++) {
                KeyType key = mergedEntries[i].first;
                ValueType value = mergedEntries[i].second;
                splitTaskQueue->pushTask([this, key, value] () {
                    this->insertKeyValue(key, value);
                });
            }  
        }
    }

    void waitForCurrentBatch() {
        for (auto& future : currentBatchTasks) {
            future.wait();
        }
        currentBatchTasks.clear();
    }

    vector<KeyType> getSegmentStartKeys() {
        return segment_start_keys;
    }

    // Get the index size in terms of bytes
    size_t get_index_size() const{
        size_t index_size = 0;

        // Size of redundant array (search index)
        index_size += redundantArray.size() * sizeof(int32_t);

        // Size of segment_start_keys
        index_size += segment_start_keys.size() * sizeof(KeyType);

        // Size of segment metadata (not including actual key-value pairs)
        for (const auto& segment : segments){
            // Segment bounds and metadata
            index_size += sizeof(KeyType) * 2;                  // lower_bound and upper_bound
            index_size += sizeof(size_t);                       // box_key_range
            index_size += sizeof(int);                          // numBoxes
            index_size += sizeof(std::atomic<int>) * 2;         // version and versionForBox

            // Box metadata (not including the key-value pairs)
            index_size += segment.getBoxCount() * sizeof(size_t);               // currentSize for each box
            index_size += segment.getBoxCount() * sizeof(std::atomic<int>);     // boxVersion for each box
            index_size += segment.getBoxCount() * sizeof(std::unique_ptr<InnerOverflowArray<KeyType, ValueType>>);     // sharedOverflow for each box
        }

        return index_size;
    }

    // Get the total size in terms of bytes (index size + sizeof(key+value))
    size_t get_total_size() const {
        // Start with the index size
        size_t size = get_index_size();
        std::cout << "index_size: " << size << std::endl;

        // Add the size of actual key-value pairs and their associated structures
        size_t total_boxes_count = 0;
        size_t total_overflow_boxes_count = 0;
        for(const auto& segment : segments){
            total_boxes_count += segment.getBoxCount();
            for (const auto& box : segment.boxes){
                size_t overflow_count = box.getOverflowBoxCount();
                total_overflow_boxes_count += overflow_count;
                // Size of main box keys and values
                size += maxKey * sizeof(KeyType);             // keys
                size += maxKey * sizeof(uint8_t);             // keys_low
                size += maxKey * sizeof(ValueType);           // values

                // Overhead for overflow structures
                if(overflow_count > 0){
                    size += sizeof(InnerOverflowArray<KeyType, ValueType>);
                    size += overflow_count * sizeof(OverflowKeyValue<KeyType, ValueType>);
                }
                /* if(overflow_count >= 2){
                    std::cout << "[Debug] overflow box count: " << overflow_count << std::endl;
                } */
            }
        }

        // Debug: check how many boxes and overflow-boxes
        std::cout << "[Debug] total boxes count: " << total_boxes_count << "; total overflow boxes count: " << total_overflow_boxes_count << std::endl;
        std::cout << "[Debug] Size of Box: " << sizeof(Box<KeyType, ValueType>) << std::endl;
        std::cout << "[Debug] Size of InnerOverflow: " << sizeof(InnerOverflowArray<KeyType, ValueType>) << std::endl;
        std::cout << "[Debug] Size of Overflow: " << sizeof(OverflowKeyValue<KeyType, ValueType>) << std::endl;
        std::cout << "[Debug] Size of sharedOverflow: " << sizeof(std::unique_ptr<InnerOverflowArray<KeyType, ValueType>>) << std::endl;
        return size;
    }
};
} 