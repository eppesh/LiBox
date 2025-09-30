#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>

int max_search_distance = 2000;
int max_actual_search_distance = 0;  // Track the maximum search distance actually used

// Global mutex for thread-safe access to generated_values
std::mutex generated_values_mutex;

// Global mutex for thread-safe printing
std::mutex print_mutex;

// Simple Thread Pool implementation
class ThreadPool {
public:
    ThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

/**
 * Generate Zipfian distribution numbers in a given range
 *
 * @param min_val Minimum value in the range (inclusive)
 * @param max_val Maximum value in the range (inclusive)
 * @param count Number of numbers to generate
 * @param s Zipfian parameter (skewness), typically between 0.5 and 2.0
 * @param output_file Path to output file
 * @param generated_values Vector to track which values have been generated (indexed by value - min_val)
 */
void generateZipfianDistribution(int64_t min_val, int64_t max_val, size_t count,
                                double s, const std::string& output_file,
                                std::vector<bool>& generated_values, bool debug = false) {
    // Validate parameters
    if (min_val >= max_val) {
        std::cerr << "Error: min_val must be less than max_val" << std::endl;
        return;
    }

    if (count == 0) {
        std::cerr << "Error: count must be greater than 0" << std::endl;
        return;
    }

    if (s <= 0) {
        std::cerr << "Error: s (skewness parameter) must be positive" << std::endl;
        return;
    }

    int64_t range_size = max_val - min_val + 1;

    // Calculate harmonic number for normalization
    double harmonic = 0.0;
    for (int64_t i = 1; i <= range_size; ++i) {
        harmonic += 1.0 / std::pow(i, s);
    }

    // Generate cumulative distribution function
    std::vector<double> cdf(range_size);
    double cumulative = 0.0;
    for (int64_t i = 0; i < range_size; ++i) {
        cumulative += 1.0 / std::pow(i + 1, s);
        cdf[i] = cumulative / harmonic;
    }

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Create mapping table with random values in the range
    std::vector<int64_t> mapping_table(range_size);
    std::uniform_int_distribution<int64_t> value_dis(min_val, max_val);

    if (debug) {
        std::cout << "Creating mapping table with " << range_size << " random values..." << std::endl;
    }
    for (int64_t i = 0; i < range_size; ++i) {
        mapping_table[i] = value_dis(gen);
    }
    if (debug) {
        std::cout << "Mapping table created successfully." << std::endl;
    }

    // print the first 10 values of the mapping table
    std::cout << "First 10 values of the mapping table: ";
    for (int64_t i = 0; i < 10; ++i) {
        std::cout << mapping_table[i] << " ";
    }
    std::cout << std::endl;

    // Open output file
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_file << std::endl;
        return;
    }

    // Generate numbers
    std::cout << "Generating " << count << " unique Zipfian distributed numbers..." << std::endl;
    std::cout << "Range: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "Skewness parameter (s): " << s << std::endl;
    std::cout << "Output file: " << output_file << std::endl;

    size_t generated_count = 0;
    size_t attempts = 0;
    const size_t max_attempts = count * 10; // Prevent infinite loops

    while (generated_count < count && attempts < max_attempts) {
        double random_val = dis(gen);

        // Find the corresponding index using binary search on CDF
        auto it = std::lower_bound(cdf.begin(), cdf.end(), random_val);
        int64_t zipfian_index = std::distance(cdf.begin(), it);

        // Ensure index is within bounds
        if (zipfian_index >= range_size) {
            zipfian_index = range_size - 1;
        }

        // Translate Zipfian index to actual value using mapping table
        int64_t mapped_value = mapping_table[zipfian_index];
        size_t mapped_value_index = mapped_value - min_val;

        // Check if this mapped value has already been generated
        if (!generated_values[mapped_value_index]) {
            generated_values[mapped_value_index] = true;
            outfile << mapped_value << std::endl;
            generated_count++;

            // Print the mapped value only in debug mode
            if (debug) {
                std::cout << "Mapped value: " << mapped_value << " (zipfian_index: " << zipfian_index
                          << ", random_val: " << random_val << ")" << std::endl;
            }

            // Progress indicator for large counts
            if (count > 100000 && generated_count % 100000 == 0) {
                std::cout << "Generated " << generated_count << " unique numbers..." << std::endl;
            }
        } else {
            // Mapped value already exists, search for neighbors near the mapped value
            if (debug) {
                std::cout << "Mapped value " << mapped_value << " already taken, searching for neighbors..." << std::endl;
            }
            bool found_neighbor = false;
            int64_t neighbor_value = -1;

            // Search in both directions (up to max_search_distance positions each way)
            for (int64_t offset = 1; offset <= max_search_distance && !found_neighbor; ++offset) {
                // Check positive offset
                int64_t pos_neighbor = mapped_value + offset;
                if (pos_neighbor <= max_val) {
                    size_t pos_index = pos_neighbor - min_val;
                    if (!generated_values[pos_index]) {
                        neighbor_value = pos_neighbor;
                        found_neighbor = true;
                        // Update max actual search distance
                        if (offset > max_actual_search_distance) {
                            max_actual_search_distance = offset;
                        }
                        break;
                    }
                }

                // Check negative offset
                int64_t neg_neighbor = mapped_value - offset;
                if (neg_neighbor >= min_val) {
                    size_t neg_index = neg_neighbor - min_val;
                    if (!generated_values[neg_index]) {
                        neighbor_value = neg_neighbor;
                        found_neighbor = true;
                        // Update max actual search distance
                        if (offset > max_actual_search_distance) {
                            max_actual_search_distance = offset;
                        }
                        break;
                    }
                }
            }

            if (found_neighbor) {
                // Use the found neighbor
                size_t neighbor_index = neighbor_value - min_val;
                generated_values[neighbor_index] = true;
                outfile << neighbor_value << std::endl;
                generated_count++;

                // Print the mapped neighbor value only in debug mode
                if (debug) {
                    std::cout << "Mapped neighbor value: " << neighbor_value << " (original_mapped: " << mapped_value
                              << ", zipfian_index: " << zipfian_index << ", random_val: " << random_val << ")" << std::endl;
                }

                // Progress indicator for large counts
                if (count > 100000 && generated_count % 100000 == 0) {
                    std::cout << "Generated " << generated_count << " unique numbers..." << std::endl;
                }
            } else {
                // No neighbors available, report error and stop
                std::cerr << "Error: No available neighbors found for mapped value " << mapped_value
                          << " within " << max_search_distance << " positions. Stopping generation." << std::endl;
                std::cerr << "Generated " << generated_count << " unique numbers before stopping." << std::endl;
                outfile.close();
                return;
            }
        }

        attempts++;
    }

    if (generated_count < count) {
        std::cerr << "Warning: Only generated " << generated_count << " unique numbers out of "
                  << count << " requested. The range might be too small for the requested count." << std::endl;
    }

    outfile.close();
    std::cout << "Successfully generated " << generated_count << " unique numbers and saved to " << output_file << std::endl;
}

/**
 * Generate Zipfian distribution numbers in batches and store in a vector
 *
 * @param min_val Minimum value in the range (inclusive)
 * @param max_val Maximum value in the range (inclusive)
 * @param count Number of numbers to generate per batch
 * @param s Zipfian parameter (skewness), typically between 0.5 and 2.0
 * @param num_batches Number of batches to generate
 * @param output_vector Vector to store the generated numbers
 * @param offset Starting offset in the output vector to store results
 * @param generated_values Vector to track which values have been generated (indexed by value - min_val)
 * @param debug Whether to enable debug output
 */
void GenZipfBatch(int64_t min_val, int64_t max_val, size_t count, double s,
                  size_t num_batches, std::vector<int64_t>& output_vector,
                  size_t offset, std::vector<bool>& generated_values,
                  const std::vector<double>& shared_cdf,
                  bool debug = false, int thread_id = -1) {
    // Validate parameters
    if (min_val >= max_val) {
        std::cerr << "Error: min_val must be less than max_val" << std::endl;
        return;
    }

    if (count == 0) {
        std::cerr << "Error: count must be greater than 0" << std::endl;
        return;
    }

    if (s <= 0) {
        std::cerr << "Error: s (skewness parameter) must be positive" << std::endl;
        return;
    }

    if (num_batches == 0) {
        std::cerr << "Error: num_batches must be greater than 0" << std::endl;
        return;
    }

    // Check if output vector has enough space
    if (output_vector.size() < offset + count * num_batches) {
        std::cerr << "Error: output_vector is too small. Need at least "
                  << offset + count * num_batches << " elements, but vector has "
                  << output_vector.size() << " elements" << std::endl;
        return;
    }

    int64_t range_size = max_val - min_val + 1;

    // Use shared CDF instead of calculating our own
    const std::vector<double>& cdf = shared_cdf;

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Create mapping table with random values in the range
    std::vector<int64_t> mapping_table(range_size);
    std::uniform_int_distribution<int64_t> value_dis(min_val, max_val);

    if (debug) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "[T" << thread_id << "] Creating mapping table with " << range_size << " random values..." << std::endl;
    }
    for (int64_t i = 0; i < range_size; ++i) {
        mapping_table[i] = value_dis(gen);
    }
    
    // Always print mapping table completion message
    {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "[T" << thread_id << "] Mapping table created successfully with " << range_size << " random values." << std::endl;
    }

    {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "[T" << thread_id << "] Generating " << num_batches << " batches of " << count
                  << " unique Zipfian distributed numbers each..." << std::endl;
        std::cout << "[T" << thread_id << "] Range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "[T" << thread_id << "] Skewness parameter (s): " << s << std::endl;
        std::cout << "[T" << thread_id << "] Total numbers to generate: " << count * num_batches << std::endl;
    }

    size_t total_generated = 0;
    const size_t max_attempts = count * 10; // Prevent infinite loops per batch

    for (size_t batch = 0; batch < num_batches; ++batch) {
        if (debug) {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cout << "[T" << thread_id << "] Starting batch " << (batch + 1) << "/" << num_batches << std::endl;
        }

        size_t generated_count = 0;
        size_t attempts = 0;
        size_t batch_offset = offset + batch * count;

        while (generated_count < count && attempts < max_attempts) {
            double random_val = dis(gen);

            // Find the corresponding index using binary search on CDF
            auto it = std::lower_bound(cdf.begin(), cdf.end(), random_val);
            int64_t zipfian_index = std::distance(cdf.begin(), it);

            // Ensure index is within bounds
            if (zipfian_index >= range_size) {
                zipfian_index = range_size - 1;
            }

            // Translate Zipfian index to actual value using mapping table
            int64_t mapped_value = mapping_table[zipfian_index];
            size_t mapped_value_index = mapped_value - min_val;

            // Check if this mapped value has already been generated (thread-safe)
            bool value_available = false;
            {
                std::lock_guard<std::mutex> lock(generated_values_mutex);
                if (!generated_values[mapped_value_index]) {
                    generated_values[mapped_value_index] = true;
                    value_available = true;
                }
            }

            if (value_available) {
                output_vector[batch_offset + generated_count] = mapped_value;
                generated_count++;

                if (debug) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << "[T" << thread_id << "] Batch " << (batch + 1) << " - Mapped value: " << mapped_value
                              << " (zipfian_index: " << zipfian_index
                              << ", random_val: " << random_val << ")" << std::endl;
                }
            } else {
                // Mapped value already exists, search for neighbors near the mapped value
                if (debug) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << "[T" << thread_id << "] Batch " << (batch + 1) << " - Mapped value " << mapped_value
                              << " already taken, searching for neighbors..." << std::endl;
                }
                bool found_neighbor = false;
                int64_t neighbor_value = -1;

                // Search in both directions (up to max_search_distance positions each way)
                for (int64_t offset_search = 1; offset_search <= max_search_distance && !found_neighbor; ++offset_search) {
                    // Check positive offset
                    int64_t pos_neighbor = mapped_value + offset_search;
                    if (pos_neighbor <= max_val) {
                        size_t pos_index = pos_neighbor - min_val;
                        bool pos_available = false;
                        {
                            std::lock_guard<std::mutex> lock(generated_values_mutex);
                            if (!generated_values[pos_index]) {
                                generated_values[pos_index] = true;
                                pos_available = true;
                            }
                        }
                        if (pos_available) {
                            neighbor_value = pos_neighbor;
                            found_neighbor = true;
                            // Update max actual search distance (thread-safe)
                            static std::mutex max_distance_mutex;
                            {
                                std::lock_guard<std::mutex> lock(max_distance_mutex);
                                if (offset_search > max_actual_search_distance) {
                                    max_actual_search_distance = offset_search;
                                }
                            }
                            break;
                        }
                    }

                    // Check negative offset
                    int64_t neg_neighbor = mapped_value - offset_search;
                    if (neg_neighbor >= min_val) {
                        size_t neg_index = neg_neighbor - min_val;
                        bool neg_available = false;
                        {
                            std::lock_guard<std::mutex> lock(generated_values_mutex);
                            if (!generated_values[neg_index]) {
                                generated_values[neg_index] = true;
                                neg_available = true;
                            }
                        }
                        if (neg_available) {
                            neighbor_value = neg_neighbor;
                            found_neighbor = true;
                            // Update max actual search distance (thread-safe)
                            static std::mutex max_distance_mutex;
                            {
                                std::lock_guard<std::mutex> lock(max_distance_mutex);
                                if (offset_search > max_actual_search_distance) {
                                    max_actual_search_distance = offset_search;
                                }
                            }
                            break;
                        }
                    }
                }

                if (found_neighbor) {
                    // Use the found neighbor (already marked as generated in search loop)
                    output_vector[batch_offset + generated_count] = neighbor_value;
                    generated_count++;

                    if (debug) {
                        std::lock_guard<std::mutex> lock(print_mutex);
                        std::cout << "[T" << thread_id << "] Batch " << (batch + 1) << " - Mapped neighbor value: " << neighbor_value
                                  << " (original_mapped: " << mapped_value
                                  << ", zipfian_index: " << zipfian_index
                                  << ", random_val: " << random_val << ")" << std::endl;
                    }
                } else {
                    // No neighbors available, report error and stop this batch
                    {
                        std::lock_guard<std::mutex> lock(print_mutex);
                        std::cerr << "[T" << thread_id << "] Error: No available neighbors found for mapped value " << mapped_value
                                  << " within " << max_search_distance << " positions in batch " << (batch + 1)
                                  << ". Stopping this batch." << std::endl;
                        std::cerr << "[T" << thread_id << "] Generated " << generated_count << " unique numbers in batch " << (batch + 1)
                                  << " before stopping." << std::endl;
                    }
                    break;
                }
            }

            attempts++;
        }

        if (generated_count < count) {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cerr << "[T" << thread_id << "] Warning: Only generated " << generated_count << " unique numbers out of "
                      << count << " requested in batch " << (batch + 1)
                      << ". The range might be too small for the requested count." << std::endl;
        }

        total_generated += generated_count;

        if (debug) {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cout << "[T" << thread_id << "] Completed batch " << (batch + 1) << "/" << num_batches
                      << " - Generated " << generated_count << " numbers" << std::endl;
        }
    }

    {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "[T" << thread_id << "] Successfully generated " << total_generated << " unique numbers across "
                  << num_batches << " batches" << std::endl;
    }
}

int main() {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Example usage with different parameters
    std::cout << "=== Zipfian Distribution Generator ===" << std::endl;

    // Generate a worst-case scenario with high skewness
    // This creates a distribution where a few values are very frequent
    // and most values are rare - challenging for learned indexes

    int64_t max_val = 1800 * 1000 * 1000ll;
    int64_t min_val = -max_val;
    double s = 0.75;             // High skewness (typical range: 0.5-2.0)
    std::string output_file = "worst_case_zipfian-100k.txt";

    // Create vector to track generated values
    int64_t range_size = max_val - min_val + 1;
    std::vector<bool> generated_values(range_size, false);

    // size_t count = 1000 * 1000;      // Generate 1 million unique numbers
    // generateZipfianDistribution(min_val, max_val, count, s, output_file, generated_values, false);

    // Concurrent usage of GenZipfBatch function with thread pool
    std::cout << "\n=== Concurrent GenZipfBatch with Thread Pool ===" << std::endl;

    // Parameters for concurrent generation
    size_t batch_count = 100000;  // Numbers per batch
    size_t num_threads = 6; // std::thread::hardware_concurrency();  // Use all available CPU cores
    size_t num_batches = 2000;  // One batch per thread
    size_t total_batch_numbers = batch_count * num_batches;

    std::cout << "Using " << num_threads << " threads for concurrent generation" << std::endl;
    std::cout << "Each thread will generate " << batch_count << " numbers" << std::endl;
    std::cout << "Total numbers to generate: " << total_batch_numbers << std::endl;

    // Create output vector to store all results
    std::vector<int64_t> batch_output(total_batch_numbers);

    // Create shared generated_values vector for all threads
    std::vector<bool> batch_generated_values(range_size, false);

    // Create shared CDF for all threads (same calculation for all threads)
    std::cout << "Creating shared CDF for all threads..." << std::endl;
    std::vector<double> shared_cdf(range_size);
    double harmonic = 0.0;
    for (int64_t i = 1; i <= range_size; ++i) {
        harmonic += 1.0 / std::pow(i, s);
    }
    double cumulative = 0.0;
    for (int64_t i = 0; i < range_size; ++i) {
        cumulative += 1.0 / std::pow(i + 1, s);
        shared_cdf[i] = cumulative / harmonic;
    }
    std::cout << "Shared CDF created successfully with " << range_size << " values." << std::endl;

    // Create thread pool and submit tasks
    {
        ThreadPool pool(num_threads);

        // Submit tasks to thread pool
        std::cout << "Submitting tasks to thread pool..." << std::endl;
        for (size_t i = 0; i < num_batches; ++i) {
            size_t offset = i * batch_count;
            pool.enqueue([&, i, offset]() {
                {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << "Thread " << i << " starting batch generation at offset " << offset << std::endl;
                }
                GenZipfBatch(min_val, max_val, batch_count, s, 1,
                             batch_output, offset, batch_generated_values, shared_cdf, false, i);
                {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << "Thread " << i << " completed batch generation" << std::endl;
                }
            });
        }

        // Wait for all tasks to complete (ThreadPool destructor will join all threads)
        std::cout << "Waiting for all threads to complete..." << std::endl;
    } // ThreadPool destructor called here, ensuring all threads are joined

    std::cout << "All threads completed. Now writing to file..." << std::endl;

    // Now write all generated numbers to output file
    std::cout << "Writing all generated numbers to output file..." << std::endl;
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file: " << output_file << std::endl;
        return 1;
    }

    size_t total_written = 0;
    for (size_t i = 0; i < total_batch_numbers; ++i) {
        outfile << batch_output[i] << std::endl;
        total_written++;
    }
    outfile.close();

    std::cout << "Successfully wrote " << total_written << " numbers to " << output_file << std::endl;

    // Print first 10 values from each batch as example
    std::cout << "\nFirst 10 values from each batch:" << std::endl;
    for (size_t batch = 0; batch < num_batches; ++batch) {
        std::cout << "Batch " << (batch + 1) << ": ";
        size_t start_idx = batch * batch_count;
        for (size_t i = 0; i < 10 && i < batch_count; ++i) {
            std::cout << batch_output[start_idx + i] << " ";
        }
        std::cout << std::endl;
    }

    // End timing and calculate duration
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Convert to minutes and seconds
    int total_seconds = duration.count() / 1000;
    int minutes = total_seconds / 60;
    int seconds = total_seconds % 60;

    // Print runtime in mm:ss format
    std::cout << "\nProgram runtime: "
              << std::setfill('0') << std::setw(2) << minutes << ":"
              << std::setfill('0') << std::setw(2) << seconds << std::endl;

    // Print maximum search distance actually used
    std::cout << "Maximum search distance used: " << max_actual_search_distance << std::endl;

    return 0;
}
