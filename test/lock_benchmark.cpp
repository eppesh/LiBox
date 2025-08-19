#include <chrono>
#include <shared_mutex>
#include <mutex>
#include <thread>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <atomic>
#include <memory>
#include <iomanip>
#include <cstdint>

class PartitionedCounter {
private:
    struct alignas(64) LocalCounter {
        std::atomic<int64_t> counter{0};
    };
    
    LocalCounter* local_counters;
    int num_counters;

public:
    PartitionedCounter(int thread_num) : num_counters(thread_num) {
        local_counters = new LocalCounter[num_counters];
    }
    
    ~PartitionedCounter() {
        delete[] local_counters;
    }
    
    int64_t get() const {
        int64_t total = 0;
        for (int i = 0; i < num_counters; i++) {
            total += local_counters[i].counter.load();
        }
        return total;
    }
    
    void add(int64_t count, uint8_t counter_id) {
        counter_id = counter_id % num_counters;
        local_counters[counter_id].counter += count;
    }
};

class ThreadLocalLock {
private:
    PartitionedCounter readers_;
    std::atomic_flag writer_;
    int num_counters;

    int get_thread_partition() const {
        return 0; // 单线程测试，总是返回0
    }

public:
    ThreadLocalLock(int thread_num) : readers_(thread_num), num_counters(thread_num) {
        writer_.clear();
    }
    
    void lock_shared() {
        int partition = get_thread_partition();
        readers_.add(1, partition);
        
        while (writer_.test(std::memory_order_acquire)) {
            readers_.add(-1, partition);
            writer_.wait(true, std::memory_order_acquire);
            readers_.add(1, partition);
        }
    }
    
    void unlock_shared() {
        int partition = get_thread_partition();
        readers_.add(-1, partition);
    }
    
    void lock() {
        while (writer_.test_and_set(std::memory_order_acq_rel)) {
            writer_.wait(true, std::memory_order_acq_rel);
        }
        
        while (readers_.get() > 0) {
        }
    }
    
    void unlock() {
        writer_.clear(std::memory_order_release);
        writer_.notify_all();
    }
    
    bool try_upgrade() {
        int partition = get_thread_partition();
        
        if (writer_.test_and_set(std::memory_order_acq_rel)) {
            readers_.add(-1, partition);
            return false;
        }
        
        readers_.add(-1, partition);
        
        while (readers_.get() > 0) {
        }
        
        return true;
    }
};

class OptimisticLock {
private:
    mutable std::atomic<uint32_t> version_lock_{0};
    static constexpr uint32_t WRITE_LOCK_BIT = 0x80000000;
    static constexpr uint32_t VERSION_MASK   = 0x7FFFFFFF;
    mutable std::atomic<bool> segment_splitting_{false};

    mutable std::atomic<uint32_t> global_split_version_{0};
    static constexpr uint32_t SPLIT_IN_PROGRESS = 0x80000000;

    volatile int data_value_ = 42;

public:
    OptimisticLock() = default;
    
    bool is_segment_splitting() const {
        return segment_splitting_.load(std::memory_order_acquire);
    }

    bool is_global_splitting() const {
        return (global_split_version_.load(std::memory_order_acquire) & SPLIT_IN_PROGRESS) != 0;
    }

    int optimistic_read() const {
        while (true) {
            uint32_t version_start;
            if (is_global_splitting()) {
                std::this_thread::yield();
                continue;
            }
            if (is_segment_splitting()) {
                std::this_thread::yield();
                continue;
            }
            if (test_write_lock(version_start)) {
                std::this_thread::yield();
                continue;
            }
            int result = data_value_;
            
            if (!version_changed(version_start)) {
                return result;
            }
        }
    }
    
    void optimistic_write(int new_value) {
        if (is_global_splitting()) {
            std::this_thread::yield();
            std::abort();
        }
        if (is_segment_splitting()) {
            std::this_thread::yield();
            std::abort();
        }
        if (!try_acquire_write_lock()) {
            std::this_thread::yield();
            return optimistic_write(new_value);
        }
        
        data_value_ = new_value;
        
        release_write_lock();
    }
    
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
};

class LockBenchmark {
private:
    static constexpr int ITERATIONS = 10000000;
    
    ThreadLocalLock libox_lock_;
    ThreadLocalLock segment_lock_;
    std::shared_mutex box_lock_;
    OptimisticLock optimistic_lock_;

    mutable std::atomic<uint32_t> global_split_version_{0};
    
    volatile int dummy_data_ = 0;
    
    void simple_operation() {
        dummy_data_ = dummy_data_ + 1;
    }
    
    int simple_read_operation() {
        return dummy_data_;
    }

public:
    LockBenchmark() : libox_lock_(1), segment_lock_(1) {}
    
    double benchmark_lockfree_operation() {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < ITERATIONS; i++) {
            simple_operation();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return static_cast<double>(duration.count()) / ITERATIONS;
    }
    
    double benchmark_lockfree_read() {
        auto start = std::chrono::high_resolution_clock::now();
        
        volatile int sum = 0;
        for (int i = 0; i < ITERATIONS; i++) {
            sum += simple_read_operation();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return static_cast<double>(duration.count()) / ITERATIONS;
    }
    
    double benchmark_optimistic_read() {
        auto start = std::chrono::high_resolution_clock::now();
        
        volatile int sum = 0;
        for (int i = 0; i < ITERATIONS; i++) {
            sum += optimistic_lock_.optimistic_read();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return static_cast<double>(duration.count()) / ITERATIONS;
    }
    
    double benchmark_optimistic_write() {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < ITERATIONS; i++) {
            optimistic_lock_.optimistic_write(i);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return static_cast<double>(duration.count()) / ITERATIONS;
    }
    
    double benchmark_optimistic_mixed() {
        auto start = std::chrono::high_resolution_clock::now();
        
        volatile int sum = 0;
        for (int i = 0; i < ITERATIONS; i++) {
            if (i % 2 == 0) {
                optimistic_lock_.optimistic_write(i);
            } else {
                sum += optimistic_lock_.optimistic_read();
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return static_cast<double>(duration.count()) / ITERATIONS;
    }
    
    double benchmark_threadlocal_shared_read() {
        auto start = std::chrono::high_resolution_clock::now();
        
        volatile int sum = 0;
        for (int i = 0; i < ITERATIONS; i++) {
            libox_lock_.lock_shared();
            std::shared_lock<std::shared_mutex> lock(box_lock_);
            
            sum += simple_read_operation();
            libox_lock_.unlock_shared();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return static_cast<double>(duration.count()) / ITERATIONS;
    }
    
    double benchmark_threadlocal_exclusive_write() {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < ITERATIONS; i++) {
            libox_lock_.lock_shared(); 
            segment_lock_.lock_shared(); 
            std::unique_lock<std::shared_mutex> lock(box_lock_);
            
            simple_operation();

            segment_lock_.unlock_shared();
            libox_lock_.unlock_shared();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return static_cast<double>(duration.count()) / ITERATIONS;
    }
    
    double benchmark_threadlocal_mixed() {
        auto start = std::chrono::high_resolution_clock::now();
        
        volatile int sum = 0;
        for (int i = 0; i < ITERATIONS; i++) {
            if (i % 2 == 0) {
                libox_lock_.lock_shared();
                std::shared_lock<std::shared_mutex> lock(box_lock_);
                
                sum += simple_read_operation();
                libox_lock_.unlock_shared();
            } else {
                libox_lock_.lock_shared();
                segment_lock_.lock_shared();
                std::unique_lock<std::shared_mutex> lock(box_lock_);
                
                simple_operation();
                segment_lock_.unlock_shared();
                libox_lock_.unlock_shared();
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return static_cast<double>(duration.count()) / ITERATIONS;
    }
    
    void run_comparison() {
        std::cout << "=== Lock Performance Benchmark ===" << std::endl;
        std::cout << "Iterations: " << ITERATIONS << std::endl;
        std::cout << "Single-threaded, zero contention test" << std::endl;
        std::cout << std::endl;
        
        benchmark_lockfree_operation();
        benchmark_optimistic_read();
        
        double lockfree_time = benchmark_lockfree_operation();
        double lockfree_read_time = benchmark_lockfree_read();
        double optimistic_read_time = benchmark_optimistic_read();
        double optimistic_write_time = benchmark_optimistic_write();
        double optimistic_mixed_time = benchmark_optimistic_mixed();
        double threadlocal_read_time = benchmark_threadlocal_shared_read();
        double threadlocal_write_time = benchmark_threadlocal_exclusive_write();
        double threadlocal_mixed_time = benchmark_threadlocal_mixed();
        
        std::cout << std::fixed << std::setprecision(2);
        
        std::cout << "=== Results (per operation) ===" << std::endl;
        std::cout << "Lock-Free Operation:             " << lockfree_time << " ns" << std::endl;
        std::cout << "Lock-Free Read:                  " << lockfree_read_time << " ns" << std::endl;
        std::cout << "Optimistic Read:                 " << optimistic_read_time << " ns" << std::endl;
        std::cout << "Optimistic Write:                " << optimistic_write_time << " ns" << std::endl;
        std::cout << "Optimistic Mixed (50% read):     " << optimistic_mixed_time << " ns" << std::endl;
        std::cout << "ThreadLocal+Shared Read:         " << threadlocal_read_time << " ns" << std::endl;
        std::cout << "ThreadLocal+Exclusive Write:     " << threadlocal_write_time << " ns" << std::endl;
        std::cout << "ThreadLocal Mixed (50/50):       " << threadlocal_mixed_time << " ns" << std::endl;
        std::cout << std::endl;
        
        std::cout << "=== Performance Ratios ===" << std::endl;
        std::cout << "Optimistic/Lock-Free Read:       " << (optimistic_read_time / lockfree_read_time) << "x" << std::endl;
        std::cout << "ThreadLocal/Lock-Free Read:      " << (threadlocal_read_time / lockfree_read_time) << "x" << std::endl;
        std::cout << "Optimistic/ThreadLocal Read:     " << (optimistic_read_time / threadlocal_read_time) << "x" << std::endl;
        std::cout << "Optimistic/Lock-Free Write:      " << (optimistic_write_time / lockfree_time) << "x" << std::endl;
        std::cout << "ThreadLocal/Lock-Free Write:     " << (threadlocal_write_time / lockfree_time) << "x" << std::endl;
        std::cout << "Optimistic/ThreadLocal Write:    " << (optimistic_write_time / threadlocal_write_time) << "x" << std::endl;
        std::cout << "Optimistic/ThreadLocal Mixed:    " << (optimistic_mixed_time / threadlocal_mixed_time) << "x" << std::endl;
    }
};

int main() {
    LockBenchmark benchmark;
    benchmark.run_comparison();
    
    return 0;
}