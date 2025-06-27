#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "graph_utils.h"

using namespace std;
using namespace liboxns;

using KeyType = int64_t;

// input: where to start (which box percentage-wise), where to end, list of window sizes
// output: csv file with window size, length of segment, ratio of underflow, ratio of overflow
// for each window size: finds the starting box, then forms segments (with the given window
// size) of increasing size until we reach the ending box (which is also a percentage). for each
// segment, calculate the ratio of underflow and overflow.
void calc_graph(vector<KeyType>& data,
                double start_percent,
                double end_percent,
                const vector<KeyType>& window_sizes,
                const string& output_file) {
    ofstream outfile(output_file);
    if (!outfile.is_open()) {
        throw runtime_error("Could not open output file: " + output_file);
    }
    FileGuard guard(outfile);

    // csv header
    outfile << "window_size,segment_length,num_keys,underflow_ratio,overflow_ratio\n";

    processData(data, start_percent, end_percent);

    for (KeyType window_size : window_sizes) {
        vector<Block<KeyType>> blocks = computeBlocksByWind(data, window_size);

        for (KeyType i = 0; i < blocks.size(); i++) {
            StructSegment<KeyType> seg =
                createSegment(blocks, static_cast<KeyType>(0), i, window_size);

            double underflow_ratio = computeUnderflowRatioAccurate(data, seg);
            double overflow_ratio = computeOverflowRatioNoUpperBound(data, seg);

            /*if (overflow_ratio > 0.1 || underflow_ratio > 0.5) {
                return;
            }*/ // DEBUG

            outfile << window_size << "," << i + 1 << ","
                    << countKeysInInterval(data, data.begin(), seg.seg_upper) << ','
                    << underflow_ratio << "," << overflow_ratio << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        cerr << "Usage: " << argv[0]
             << " <input_file> <start_percent> <end_percent> <window_start> <window_end> "
                "<increment>"
             << endl;
        return 1;
    }

    string input_file = argv[1];
    double start_percent = stod(argv[2]);
    double end_percent = stod(argv[3]);
    KeyType window_start = stoul(argv[4]);
    KeyType window_end = stoul(argv[5]);
    KeyType increment = stoul(argv[6]);

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

    string output_file = "len" + toScientificNotation(data.size()) + "_p" +
                         toScientificNotation(start_percent * 100) + "t" +
                         toScientificNotation(end_percent * 100) + "_r" +
                         toScientificNotation(window_start) + "t" +
                         toScientificNotation(window_end) + ".csv";

    try {
        calc_graph(data, start_percent, end_percent, window_sizes, output_file);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
