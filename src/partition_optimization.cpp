#include <fstream>
#include <iostream>
#include <sstream>

#include "segmentation.h"

using namespace liboxns;

using KeyType = int64_t; // vmware&cambridge uint32_t; longitudes-200M int64_T

bool load_data(const std::string& input, std::vector<KeyType>& data) {
    std::ifstream fin(input);
    if (!fin) {
        std::cerr << "Cannot open input file: " << input << endl;
        return false;
    }
    data.clear();

    std::string line = "";
    if (input.find("fiu") != std::string::npos) {
        while (std::getline(fin, line)) {
            std::istringstream iss(line);
            std::string token;
            std::getline(iss, token, ',');
            data.push_back(std::stoull(token));
        }
    } else if (input.find("umass") != std::string::npos ||
               input.find(".csv") != std::string::npos || input.find(".txt") != std::string::npos) {
        while (getline(fin, line)) {
            if (line.empty()) continue;
            // data.push_back(std::stoull(line));
            if constexpr (std::is_same_v<KeyType, double>) {
                data.push_back(std::stod(line));
            } else if constexpr (std::is_signed_v<KeyType>) {
                data.push_back(std::stoll(line));
            } else {
                data.push_back(std::stoull(line));
            }
        }
    } else { // binary
        std::cout << "read binary file..." << std::endl;
        std::ifstream infile(input, std::ios::in | std::ios_base::binary);
        if (!infile.is_open()) {
            std::cout << "[load_data] Error opening " << input << std::endl;
            return false;
        }
        if (input.find("covid") != std::string::npos || input.find("genome") != std::string::npos ||
            input.find("fb") != std::string::npos || input.find("osm") != std::string::npos) {
            uint64_t count; // The first 8 bytes is for max size in gre traces.
            infile.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));
            std::cout << "gre max size = " << count << std::endl;
        }
        KeyType key;
        while (!infile.read(reinterpret_cast<char*>(&key), sizeof(KeyType)).eof()) {
            data.push_back(key);
        }
        infile.close();
    }
    fin.close();
    return true;
}

// void process_data_shrink(std::vector<KeyType>& data) {
//     sort(data.begin(), data.end());
//     data.erase(unique(data.begin(), data.end()), data.end());

//     vector<KeyType> shrunk_data;
//     for (size_t i = 0; i < data.size(); i += SHRINK_FACTOR) {
//         shrunk_data.push_back(data[i]);
//     }
//     data = std::move(shrunk_data);
// }

void process_data(std::vector<KeyType>& data) {
    sort(data.begin(), data.end());
    data.erase(unique(data.begin(), data.end()), data.end());
}

int main(int argc, char* argv[]) {
    // Modify the 'KeyType' as needed
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        std::cerr << "E.g.: ./partition_optimization w106.csv optimized_segments_w106.csv"
                  << std::endl;
        return -1;
    }
    std::cout << "Size of current KeyType is " << sizeof(KeyType) << std::endl;
    string input_file = argv[1];
    string output_file = argv[2];

    vector<KeyType> data;
    if (!load_data(input_file, data)) {
        std::cerr << "Load data failed!" << std::endl;
        return -1;
    }
    std::cout << "Finish loading the data. Original data size=" << data.size() << std::endl;

    // SHRINK!
    process_data(data);
    // process_data_shrink(data);
    std::cout << "data size=" << data.size() << ", min=" << data.front() << ", max=" << data.back()
              << std::endl;

    // SHRINK!
    int max_look_ahead = 15;
    double underflow_threshold = 0.5;
    double overflow_threshold = 0.1;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<Segment<KeyType>> segments =
        calculateSegments(data, overflow_threshold, underflow_threshold, max_look_ahead);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::vector<StructSegment<KeyType>> final_segments = toStructSegment(segments);

    std::cout << "Finished creating segments: segment count=" << final_segments.size() << std::endl;
    if (!final_segments.empty()) {
        std::cout << "last segment: lower=" << final_segments.back().seg_lower
                  << "; upper=" << final_segments.back().seg_upper 
                  << "; range=" << final_segments.back().box_range << std::endl;
    }

    auto duration_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                                       start_time);
    auto duration_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                                      start_time);
    std::cout << "calculateSegments execution time: " << duration_microseconds.count()
              << " microseconds (" << duration_seconds.count() << " seconds)" << std::endl;

#ifndef NDEBUG
    bool valid = validateSegments(data, segments);
    if (valid) {
        std::cout << "Confirmed that the segments are valid!" << std::endl;
    } else {
        std::cout << "ERROR/WARNING THE SEGMENTS ARE NOT VALID" << std::endl;
    }
#endif

    ofstream fout(output_file);
    if (!fout) {
        cerr << "Cannot open output file: " << output_file << endl;
        return 1;
    }

    KeyType start;
    KeyType end;
    for (int i = 0; i < final_segments.size(); i++) {
        if (i == 0) {
            start = final_segments[0].seg_lower;
            end = final_segments[0].seg_upper;
        } else {
            start = end;
            end = final_segments[i].seg_upper;
        }
        // SHRINK!
        fout << start << "," << end << "," << final_segments[i].box_range << "\n";
        // fout << start << "," << end << "," << final_segments[i].box_range / SHRINK_FACTOR << "\n";
    }
    fout.close();
    cout << "Segmentation results saved to " << output_file << endl;
    return 0;
}