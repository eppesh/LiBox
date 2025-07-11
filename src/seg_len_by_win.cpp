#include <algorithm>
#include <climits>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "graph_utils.h"

using namespace std;
using namespace liboxns;

const double UNDERFLOW_THRESHOLD = 0.5;
const double OVERFLOW_THRESHOLD = 0.1;

using KeyType = int64_t;

template <typename KeyType>
StructSegment<KeyType> createMaxSegment(const vector<Block<KeyType>>& blocks,
                                        const vector<KeyType>& data,
                                        const KeyType window_size,
                                        double underflowThreshold,
                                        double overflowThreshold,
                                        int maxMergeCount) {
    KeyType block_idx = 1;
    KeyType mergeCount = 0;
    while (block_idx < blocks.size() && mergeCount < maxMergeCount) {
        StructSegment<KeyType> candidate = createSegment(blocks,
                                                         static_cast<KeyType>(0),
                                                         block_idx);
        double uf = computeUnderflowRatioAccurate(data, candidate);
        double of = computeOverflowRatioNoUpperBound(data, candidate);
        if (uf <= underflowThreshold && of <= overflowThreshold) {
            mergeCount = 0;
        } else {
            mergeCount++;
        }
        block_idx++;
    }
    return createSegment(blocks, static_cast<KeyType>(0), block_idx - mergeCount - 1, window_size);
}

void calc_graph(vector<KeyType>& data,
                double start_percent,
                double end_percent,
                const vector<KeyType>& window_sizes,
                const int max_merge_count,
                const string& output_file) {
    ofstream outfile(output_file);
    if (!outfile.is_open()) {
        throw runtime_error("Could not open output file: " + output_file);
    }
    FileGuard guard(outfile);

    // csv header
    outfile << "window_size,max_segment_length,num_keys\n";

    processData(data, start_percent, end_percent);

    for (KeyType window_size : window_sizes) {
        vector<Block<KeyType>> blocks = computeBlocksByWind(data, window_size);

        StructSegment<KeyType> max_seg = createMaxSegment(
            blocks, data, window_size, UNDERFLOW_THRESHOLD, OVERFLOW_THRESHOLD, max_merge_count);

        outfile << window_size << "," << (max_seg.endIndex - max_seg.startIndex + 1) << ","
                << countKeysInInterval(data, data.begin(), max_seg.seg_upper) << endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 7 || argc > 8) {
        cerr << "Usage: " << argv[0]
             << " <input_file> <start_percent> <end_percent> <window_start> <window_end> "
                "<increment> [max_merge_count]"
             << endl;
        cerr << "  max_merge_count is optional, defaults to maximum value (no limit)" << endl;
        return 1;
    }

    string input_file = argv[1];
    double start_percent = stod(argv[2]);
    double end_percent = stod(argv[3]);
    KeyType window_start = stoul(argv[4]);
    KeyType window_end = stoul(argv[5]);
    KeyType increment = stoul(argv[6]);
    int max_merge_count = (argc == 8) ? stoi(argv[7]) : INT_MAX;

    // Generate vector of window sizes
    vector<KeyType> window_sizes;
    for (KeyType size = window_start; size <= window_end; size += increment) {
        window_sizes.push_back(size);
    }

    // Read input data
    ifstream infile(input_file);
    if (!infile.is_open()) {
        cerr << "Could not open input file: " << input_file << endl;
        return 1;
    }

    vector<KeyType> data;
    KeyType value;
    while (infile >> value) {
        data.push_back(value);
    }

    string max_merge_str = (max_merge_count == INT_MAX) ? "-1" : to_string(max_merge_count);
    string output_file = "m" + max_merge_str + "_len" + toScientificNotation(data.size()) + "_p" +
                         toScientificNotation(start_percent * 100) + "t" +
                         toScientificNotation(end_percent * 100) + "_r" +
                         toScientificNotation(window_start) + "t" +
                         toScientificNotation(window_end) + ".csv";

    try {
        calc_graph(data, start_percent, end_percent, window_sizes, max_merge_count, output_file);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
