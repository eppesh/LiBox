#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <future>
#include <algorithm>

struct Bucket {
    double start;
    double end;
    int count;
};

class KeyCounter {
private:
    std::vector<long long> keys;
    std::vector<Bucket> buckets;
    long long min_key, max_key;
    int num_buckets;

public:
    KeyCounter(long long min_key, long long max_key, int num_buckets)
        : min_key(min_key), max_key(max_key), num_buckets(num_buckets) {
        create_buckets();
    }

    void create_buckets() {
        double bucket_width = static_cast<double>(max_key - min_key) / num_buckets;
        buckets.resize(num_buckets);

        for (int i = 0; i < num_buckets; ++i) {
            buckets[i].start = min_key + i * bucket_width;
            buckets[i].end = min_key + (i + 1) * bucket_width;
            buckets[i].count = 0;
        }
    }

    void read_keys_from_file(const std::string& filename, size_t skip_lines = 0) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            exit(1);
        }

        std::string line;
        long long key;
        size_t line_count = 0;
        size_t skipped_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_progress_time = start_time;

        std::cout << "Reading keys from " << filename;
        if (skip_lines > 0) {
            std::cout << " (skipping first " << skip_lines << " lines)";
        }
        std::cout << "..." << std::endl;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            // Skip the first N lines
            if (skipped_count < skip_lines) {
                skipped_count++;
                continue;
            }

            try {
                key = std::stoll(line);
                keys.push_back(key);
                line_count++;

                // Show progress every 1 second
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_progress_time);

                if (elapsed.count() >= 1000) {
                    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                    double rate = static_cast<double>(line_count) / (total_elapsed.count() / 1000.0);
                    std::cout << "\rProgress: " << line_count << " keys read ("
                              << static_cast<int>(rate) << " keys/sec)" << std::flush;
                    last_progress_time = current_time;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Skipping invalid line: " << line << std::endl;
            }
        }

        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        double rate = static_cast<double>(line_count) / (total_time.count() / 1000.0);
        std::cout << "\rCompleted: " << line_count << " keys read in "
                  << total_time.count() / 1000.0 << "s ("
                  << static_cast<int>(rate) << " keys/sec)";
        if (skip_lines > 0) {
            std::cout << " (skipped " << skipped_count << " lines)";
        }
        std::cout << std::endl;
    }

    int find_bucket(long long key) {
        if (key < min_key || key > max_key) {
            return -1; // Out of range
        }

        double bucket_width = static_cast<double>(max_key - min_key) / num_buckets;
        int bucket_idx = static_cast<int>((key - min_key) / bucket_width);

        // Handle edge case for max_key
        if (bucket_idx >= num_buckets) {
            bucket_idx = num_buckets - 1;
        }

        return bucket_idx;
    }

    void count_keys_chunk(const std::vector<long long>& chunk, std::vector<int>& local_counts) {
        for (long long key : chunk) {
            int bucket_idx = find_bucket(key);
            if (bucket_idx >= 0) {
                local_counts[bucket_idx]++;
            }
        }
    }

    void count_keys_parallel() {
        if (keys.empty()) return;

        int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()),
                                  std::max(1, static_cast<int>(keys.size()) / 10000));

        std::cout << "Counting " << keys.size() << " keys into " << num_buckets
                  << " buckets using " << num_threads << " threads..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Split keys into chunks
        size_t chunk_size = keys.size() / num_threads;
        std::vector<std::future<std::vector<int>>> futures;

        for (int i = 0; i < num_threads; ++i) {
            size_t start_idx = i * chunk_size;
            size_t end_idx = (i == num_threads - 1) ? keys.size() : (i + 1) * chunk_size;

            std::vector<long long> chunk(keys.begin() + start_idx, keys.begin() + end_idx);

            futures.push_back(std::async(std::launch::async, [this, chunk]() {
                std::vector<int> local_counts(num_buckets, 0);
                count_keys_chunk(chunk, local_counts);
                return local_counts;
            }));
        }

        // Collect results
        for (auto& future : futures) {
            std::vector<int> local_counts = future.get();
            for (int i = 0; i < num_buckets; ++i) {
                buckets[i].count += local_counts[i];
            }
        }

        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        double rate = static_cast<double>(keys.size()) / (total_time.count() / 1000.0);

        std::cout << "Completed: " << keys.size() << " keys processed in "
                  << total_time.count() / 1000.0 << "s ("
                  << static_cast<int>(rate) << " keys/sec) using " << num_threads << " threads" << std::endl;
    }

    void write_output(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot create output file " << filename << std::endl;
            exit(1);
        }

        // Write header
        file << "bucket_start,bucket_end,bucket_center,count,cdf" << std::endl;

        // Calculate CDF
        int total_keys = 0;
        for (const auto& bucket : buckets) {
            total_keys += bucket.count;
        }

        double cumulative = 0.0;
        for (const auto& bucket : buckets) {
            double center = (bucket.start + bucket.end) / 2.0;
            cumulative += bucket.count;
            double cdf = total_keys > 0 ? cumulative / total_keys : 0.0;

            file << bucket.start << "," << bucket.end << "," << center
                 << "," << bucket.count << "," << cdf << std::endl;
        }

        std::cout << "Output written to " << filename << std::endl;
    }

    void print_statistics() {
        int total_keys = 0;
        int non_empty_buckets = 0;
        int max_count = 0;
        int max_bucket_idx = 0;

        for (int i = 0; i < num_buckets; ++i) {
            total_keys += buckets[i].count;
            if (buckets[i].count > 0) {
                non_empty_buckets++;
            }
            if (buckets[i].count > max_count) {
                max_count = buckets[i].count;
                max_bucket_idx = i;
            }
        }

        std::cout << "\n=== Key Distribution Analysis ===" << std::endl;
        std::cout << "Total keys: " << total_keys << std::endl;
        std::cout << "Min key: " << min_key << std::endl;
        std::cout << "Max key: " << max_key << std::endl;
        std::cout << "Key range: " << (max_key - min_key) << std::endl;
        std::cout << "Number of buckets: " << num_buckets << std::endl;
        std::cout << "Non-empty buckets: " << non_empty_buckets << std::endl;
        std::cout << "Bucket width: " << static_cast<double>(max_key - min_key) / num_buckets << std::endl;
        std::cout << "Bucket with most keys: " << max_bucket_idx << " (count: " << max_count << ")" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <csv_file> <min_key> <max_key> <num_buckets> [output_file] [skip_lines]" << std::endl;
        std::cerr << "Example: " << argv[0] << " data.csv -2000000000 2000000000 500" << std::endl;
        std::cerr << "Example: " << argv[0] << " data.csv -2000000000 2000000000 500 output.csv 1" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    long long min_key = std::stoll(argv[2]);
    long long max_key = std::stoll(argv[3]);
    int num_buckets = std::stoi(argv[4]);
    std::string output_file = argc > 5 ? argv[5] : "bucket_data.csv";
    size_t skip_lines = argc > 6 ? std::stoul(argv[6]) : 0;

    if (min_key >= max_key) {
        std::cerr << "Error: min_key must be less than max_key." << std::endl;
        return 1;
    }

    if (num_buckets <= 0) {
        std::cerr << "Error: num_buckets must be positive." << std::endl;
        return 1;
    }

    std::cout << "Analyzing key distribution in '" << input_file << "' with " << num_buckets << " buckets..." << std::endl;
    std::cout << "Key range: [" << min_key << ", " << max_key << "]" << std::endl;
    if (skip_lines > 0) {
        std::cout << "Skipping first " << skip_lines << " lines" << std::endl;
    }

    auto overall_start = std::chrono::high_resolution_clock::now();

    KeyCounter counter(min_key, max_key, num_buckets);
    counter.read_keys_from_file(input_file, skip_lines);
    counter.count_keys_parallel();
    counter.print_statistics();
    counter.write_output(output_file);

    auto overall_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - overall_start);

    std::cout << "\n============================================================" << std::endl;
    std::cout << "ANALYSIS COMPLETE" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Total processing time: " << overall_time.count() / 1000.0 << " seconds" << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
