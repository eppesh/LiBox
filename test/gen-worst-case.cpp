#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

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
                                std::vector<bool>& generated_values) {
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

        // Find the corresponding value using binary search
        auto it = std::lower_bound(cdf.begin(), cdf.end(), random_val);
        int64_t index = std::distance(cdf.begin(), it);

        // Ensure index is within bounds
        if (index >= range_size) {
            index = range_size - 1;
        }

        int64_t value = min_val + index;
        size_t value_index = value - min_val;

        // Check if this value has already been generated
        if (!generated_values[value_index]) {
            generated_values[value_index] = true;
            outfile << value << std::endl;
            generated_count++;

            // Progress indicator for large counts
            if (count > 100000 && generated_count % 100000 == 0) {
                std::cout << "Generated " << generated_count << " unique numbers..." << std::endl;
            }
        } else {
            // Value already exists, search for neighbors within 100 positions
            bool found_neighbor = false;
            int64_t neighbor_value = -1;

            // Search in both directions (up to 100 positions each way)
            for (int64_t offset = 1; offset <= 100 && !found_neighbor; ++offset) {
                // Check positive offset
                int64_t pos_neighbor = value + offset;
                if (pos_neighbor <= max_val) {
                    size_t pos_index = pos_neighbor - min_val;
                    if (!generated_values[pos_index]) {
                        neighbor_value = pos_neighbor;
                        found_neighbor = true;
                        break;
                    }
                }

                // Check negative offset
                int64_t neg_neighbor = value - offset;
                if (neg_neighbor >= min_val) {
                    size_t neg_index = neg_neighbor - min_val;
                    if (!generated_values[neg_index]) {
                        neighbor_value = neg_neighbor;
                        found_neighbor = true;
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

                // Progress indicator for large counts
                if (count > 100000 && generated_count % 100000 == 0) {
                    std::cout << "Generated " << generated_count << " unique numbers..." << std::endl;
                }
            } else {
                // No neighbors available, report error and stop
                std::cerr << "Error: No available neighbors found for value " << value
                          << " within 100 positions. Stopping generation." << std::endl;
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

int main() {
    // Example usage with different parameters
    std::cout << "=== Zipfian Distribution Generator ===" << std::endl;

    // Generate a worst-case scenario with high skewness
    // This creates a distribution where a few values are very frequent
    // and most values are rare - challenging for learned indexes

    int64_t min_val = -1800000000;
    int64_t max_val = 1800000000;  // 3.6 billion possible values
    size_t count = 1000000;      // Generate 1 million unique numbers
    double s = 0.8;             // High skewness (typical range: 0.5-2.0)
    std::string output_file = "worst_case_zipfian.txt";

    // Create vector to track generated values
    int64_t range_size = max_val - min_val + 1;
    std::vector<bool> generated_values(range_size, false);

    generateZipfianDistribution(min_val, max_val, count, s, output_file, generated_values);

    return 0;
}
