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
#include <deque>
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

#define maxKey 64

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

inline void exponential_backoff(int retry_count) {
    if (retry_count > 10) retry_count = 10;
    int backoff = (1 << retry_count);
    std::this_thread::sleep_for(std::chrono::microseconds(backoff));
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
class SimpleBox {
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
        nearestEmptySlot = maxKey;
    }

public:
    SimpleBox() {}
    
    InsertStatus insertKeyValue(KeyType key, ValueType value) {
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

        size_t existingIndex = findKeyIndex(key);
        if (existingIndex != maxKey) {
            values[existingIndex] = value;
            uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
            version_lock_.store(new_version, std::memory_order_release);
            return InsertStatus::SUCCESS;
        }

        if (nearestEmptySlot >= maxKey) {
            uint32_t new_version = ((expected & VERSION_MASK) + 1) & VERSION_MASK;
            version_lock_.store(new_version, std::memory_order_release);
            return InsertStatus::FULL;
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
        return InsertStatus::SUCCESS;
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
        }

        if (start_version != version_lock_.load(std::memory_order_acquire)) {
            exponential_backoff(retry_count++);
            goto retry_read;
        }
        return {status, result_value};
    }

    bool isFull() const {
        return nearestEmptySlot >= maxKey;
    }

    size_t size() const {
        return maxSize;
    }
};

template <typename KeyType, typename ValueType>
class SimpleSegment {
private:
    static constexpr size_t PHYSICAL_BOXES_PER_LOGICAL = 5;
    
    size_t box_key_range;
    size_t logical_box_count;
    size_t physical_box_count;
    
    SimpleBox<KeyType, ValueType>* first_box_ptr;
    std::deque<std::atomic<uint8_t>> logical_box_write_positions;

public:
    KeyType lower_bound; 
    KeyType upper_bound;

    SimpleSegment(KeyType lower, KeyType upper, size_t box_range)
        : lower_bound(lower), upper_bound(upper), box_key_range(box_range) {
        
        size_t total_key_range = upper - lower + 1;
        logical_box_count = total_key_range / box_range;
        if (total_key_range % box_range != 0) logical_box_count++;
        
        physical_box_count = logical_box_count * PHYSICAL_BOXES_PER_LOGICAL;
        
        first_box_ptr = new SimpleBox<KeyType, ValueType>[physical_box_count];
        for (size_t i = 0; i < physical_box_count; i++) {
            new (&first_box_ptr[i]) SimpleBox<KeyType, ValueType>();
        }
        
        logical_box_write_positions.resize(logical_box_count);
        for (size_t i = 0; i < logical_box_count; i++) {
            logical_box_write_positions[i].store(0, std::memory_order_relaxed);
        }
    }

    ~SimpleSegment() {
        delete[] first_box_ptr;
    }

    size_t getLogicalBoxIndex(KeyType key) const {
        return (key - lower_bound) / box_key_range;
    }

    size_t getPhysicalBoxIndex(size_t logical_box_index, uint8_t position_offset) const {
        return logical_box_index * PHYSICAL_BOXES_PER_LOGICAL + position_offset;
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) {
        size_t logical_box_index = getLogicalBoxIndex(key);
        uint8_t current_position = logical_box_write_positions[logical_box_index].load(std::memory_order_acquire);
        
        while (current_position < PHYSICAL_BOXES_PER_LOGICAL) {
            size_t physical_box_index = getPhysicalBoxIndex(logical_box_index, current_position);
            
            InsertStatus status = (first_box_ptr + physical_box_index)->insertKeyValue(key, value);
            
            if (status == InsertStatus::SUCCESS) {
                return {InsertStatus::SUCCESS, static_cast<int>(logical_box_index)};
            }
            
            if (status == InsertStatus::FULL) {
                uint8_t expected = current_position;
                if (logical_box_write_positions[logical_box_index].compare_exchange_weak(
                    expected, current_position + 1, 
                    std::memory_order_acq_rel, 
                    std::memory_order_acquire)) {
                    current_position++;
                } else {
                    current_position = logical_box_write_positions[logical_box_index].load(std::memory_order_acquire);
                }
            } else {
                return {status, static_cast<int>(logical_box_index)};
            }
        }
        
        return {InsertStatus::FULL, static_cast<int>(logical_box_index)};
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        if (key < lower_bound || key > upper_bound) {
            return {SearchStatus::OUT_OF_RANGE, -1};
        }

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

    size_t getLogicalBoxCount() const { return logical_box_count; }
    size_t getPhysicalBoxCount() const { return physical_box_count; }
    KeyType getLowerBound() const { return lower_bound; }
    KeyType getUpperBound() const { return upper_bound; }
};

template <typename KeyType, typename ValueType>
class LiBox {
private:
    double underflowThreshold;
    double overflowThreshold;
    int thread_num;
    std::vector<SimpleSegment<KeyType, ValueType>*> segments;
    std::vector<KeyType> segment_start_keys;
    std::vector<int32_t> redundantArray;
    double a, b;

public:
    LiBox(double uThreshold, double oThreshold, int thread_num)
        : underflowThreshold(uThreshold), overflowThreshold(oThreshold), thread_num(thread_num) {
        ThreadIdManager::initialize(thread_num);
    }

    ~LiBox() {
        for (auto* seg : segments) {
            delete seg;
        }
    }

    InsertResult insertKeyValue(KeyType key, ValueType value) {
        int32_t seg_index = searchIndex(key);
        return segments[seg_index]->insertKeyValue(key, value);
    }

    SearchResult<KeyType, ValueType> searchKey(KeyType key) {
        int32_t seg_index = searchIndex(key);
        return segments[seg_index]->searchKey(key);
    }

    void buildSearchIndex() {
        if (segment_start_keys.empty()) return;

        int64_t redundantSize = segment_start_keys.size() * 90;
        redundantArray.resize(redundantSize, -1);

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
        } else {
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

            auto* seg = new SimpleSegment<KeyType, ValueType>(lower, upper, box_range);
            segments.push_back(seg);
            segment_start_keys.push_back(lower);
        }

        if (!segments.empty()) {
            segment_start_keys.push_back(segments.back()->getUpperBound() + 1);
        }

        buildSearchIndex();
    }

    void buildIndex(vector<KeyType>* file_keys) {
        int keys_size = file_keys->size();
        int inserted = 0;
        omp_set_num_threads(thread_num);
        #pragma omp parallel for reduction(+ : inserted)
        for (int i = 0; i < keys_size; i++) {
            if (insertKeyValue((*file_keys)[i], 1).status == InsertStatus::SUCCESS) {
                inserted++;
            }
        }
        cout << "bulk loading finished, inserted " << inserted << " keys \n";
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

    void printWaitTimingStats() {
        std::cout << "\n=== Wait Timing Statistics ===" << std::endl;
        std::cout << "Simple version: No timing statistics available" << std::endl;
        std::cout << "==============================\n" << std::endl;
    }
};

}