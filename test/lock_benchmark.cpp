#include <chrono>
#include <shared_mutex>
#include <mutex>
#include <thread>
#include <iostream>
#include <vector>
#include <atomic>
#include <memory>
#include <iomanip>

class ThreadLocalLock {
private:
    struct alignas(64) ThreadLockState {
        std::atomic<int> shared_count{0};
        std::atomic<bool> exclusive_held{false};
        char padding[56];
    };
    
    std::unique_ptr<ThreadLockState[]> thread_states_;
    std::atomic<bool> global_exclusive_{false};
    int thread_num_;
    
    int get_thread_slot() const {
        return 0;
    }

public:
    explicit ThreadLocalLock(int thread_num) : thread_num_(thread_num) {
        thread_states_ = std::make_unique<ThreadLockState[]>(thread_num);
    }
    
    void lock_shared() {
        int slot = get_thread_slot();
        while (global_exclusive_.load(std::memory_order_acquire)) {
        }
        thread_states_[slot].shared_count.fetch_add(1, std::memory_order_acq_rel);
        if (global_exclusive_.load(std::memory_order_acquire)) {
            thread_states_[slot].shared_count.fetch_sub(1, std::memory_order_acq_rel);
            lock_shared();
        }
    }
    
    void unlock_shared() {
        int slot = get_thread_slot();
        thread_states_[slot].shared_count.fetch_sub(1, std::memory_order_acq_rel);
    }
    
    void lock() {
        bool expected = false;
        while (!global_exclusive_.compare_exchange_weak(expected, true, std::memory_order_acq_rel)) {
            expected = false;
        }
        for (int i = 0; i < thread_num_; i++) {
            while (thread_states_[i].shared_count.load(std::memory_order_acquire) > 0) {
            }
        }
    }
    
    void unlock() {
        global_exclusive_.store(false, std::memory_order_release);
    }
};

class OptimisticLock {
private:
    mutable std::atomic<uint32_t> version_lock_{0};
    static constexpr uint32_t WRITE_LOCK_BIT = 0x80000000;
    static constexpr uint32_t VERSION_MASK   = 0x7FFFFFFF;
    
    volatile int data_value_ = 42;

public:
    OptimisticLock() = default;
    
    int optimistic_read() const {
        while (true) {
            uint32_t version_start;
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
            } else {
                libox_lock_.lock_shared();
                segment_lock_.lock_shared();
                std::unique_lock<std::shared_mutex> lock(box_lock_);
                
                simple_operation();
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
        std::cout << "Optimistic Mixed (90% read):     " << optimistic_mixed_time << " ns" << std::endl;
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