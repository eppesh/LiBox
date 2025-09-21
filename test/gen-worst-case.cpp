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
 */
void generateZipfianDistribution(int64_t min_val, int64_t max_val, size_t count, 
                                double s, const std::string& output_file) {
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
    std::cout << "Generating " << count << " Zipfian distributed numbers..." << std::endl;
    std::cout << "Range: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "Skewness parameter (s): " << s << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    
    for (size_t i = 0; i < count; ++i) {
        double random_val = dis(gen);
        
        // Find the corresponding value using binary search
        auto it = std::lower_bound(cdf.begin(), cdf.end(), random_val);
        int64_t index = std::distance(cdf.begin(), it);
        
        // Ensure index is within bounds
        if (index >= range_size) {
            index = range_size - 1;
        }
        
        int64_t value = min_val + index;
        outfile << value << std::endl;
        
        // Progress indicator for large counts
        if (count > 100000 && (i + 1) % 100000 == 0) {
            std::cout << "Generated " << (i + 1) << " numbers..." << std::endl;
        }
    }
    
    outfile.close();
    std::cout << "Successfully generated " << count << " numbers and saved to " << output_file << std::endl;
}

int main() {
    // Example usage with different parameters
    std::cout << "=== Zipfian Distribution Generator ===" << std::endl;
    
    // Generate a worst-case scenario with high skewness
    // This creates a distribution where a few values are very frequent
    // and most values are rare - challenging for learned indexes
    
    int64_t min_val = 1;
    int64_t max_val = 1000000;  // 1 million possible values
    size_t count = 100000;      // Generate 100k numbers
    double s = 1.5;             // High skewness (typical range: 0.5-2.0)
    std::string output_file = "worst_case_zipfian.txt";
    
    generateZipfianDistribution(min_val, max_val, count, s, output_file);
    
    std::cout << "\n=== Additional Examples ===" << std::endl;
    
    // Generate a more uniform distribution (lower skewness)
    std::cout << "\nGenerating uniform-like distribution..." << std::endl;
    generateZipfianDistribution(1, 10000, 50000, 0.5, "uniform_like_zipfian.txt");
    
    // Generate an extremely skewed distribution
    std::cout << "\nGenerating extremely skewed distribution..." << std::endl;
    generateZipfianDistribution(1, 100000, 200000, 2.0, "extreme_skew_zipfian.txt");
    
    std::cout << "\nAll files generated successfully!" << std::endl;
    std::cout << "You can use these files with the LiBox benchmark tool." << std::endl;
    
    return 0;
}
