#include <fstream>
#include <iostream>
#include <sstream>

#include "libox_utils.h"

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

    sort(data.begin(), data.end());
    data.erase(unique(data.begin(), data.end()), data.end());
    std::cout << "data size=" << data.size() << ", min=" << data.front() << ", max=" << data.back()
              << std::endl;
    vector<Block<KeyType>> blocks = computeBlocks(data, BLOCK_SIZE);
    std::cout << "Compute blocks finished: block count=" << blocks.size()
              << "; lower=" << blocks.back().startKey << ", upper=" << blocks.back().endKey
              << ", range=" << blocks.back().range << std::endl;

    int maxMergeCount = 3;
    double underflowThreshold = 0.5;
    double overflowThreshold = 0.1;
    std::cout << "maxMergeCount=" << maxMergeCount << ", ufthreshold=" << underflowThreshold
              << ", ofthreshold=" << overflowThreshold << std::endl;
    vector<StructSegment<KeyType>> initSegments =
        partitionSegmentsOverall(blocks, data, underflowThreshold, maxMergeCount);
    std::cout << "initSegments finished! init size=" << initSegments.size() << std::endl;
    std::cout << "last segment: lower=" << initSegments.back().seg_lower
              << "; upper=" << initSegments.back().seg_upper
              << "; range=" << initSegments.back().box_range << std::endl;
    vector<StructSegment<KeyType>> finalSegments = expandSegments(
        initSegments, blocks, data, underflowThreshold, overflowThreshold, maxMergeCount);
    std::cout << "finalSegments finished! final size=" << finalSegments.size() << std::endl;
    std::cout << "last segment: lower=" << finalSegments.back().seg_lower
              << "; upper=" << finalSegments.back().seg_upper
              << "; range=" << finalSegments.back().box_range << std::endl;

    ofstream fout(output_file);
    if (!fout) {
        cerr << "Cannot open output file: " << output_file << endl;
        return 1;
    }

    KeyType start;
    KeyType end;
    for (int i = 0; i < finalSegments.size(); i++) {
        if (i == 0) {
            start = finalSegments[0].seg_lower;
            end = finalSegments[0].seg_upper;
        } else {
            start = end;
            end = finalSegments[i].seg_upper;
        }
        fout << start << "," << end << "," << finalSegments[i].box_range << "\n";
    }
    fout.close();
    cout << "Segmentation results saved to " << output_file << endl;
    return 0;
}