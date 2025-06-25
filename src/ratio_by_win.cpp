#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "libox_utils.h"

using namespace std;
using namespace liboxns;

// RAII wrapper for file handling
class FileGuard {
   private:
    ofstream& file;

   public:
    explicit FileGuard(ofstream& f) : file(f) {}
    ~FileGuard() {
        if (file.is_open()) {
            file.close();
        }
    }
};

using KeyType = int64_t;
KeyType KEY_MAX = std::numeric_limits<KeyType>::max();

void printBlocks(const vector<Block<KeyType>>& blocks) {
    for (const auto& block : blocks) {
        cout << "Start Key: " << block.startKey << ", End Key: " << block.endKey
             << ", Range: " << block.range << endl;
    }
}

void processData(vector<KeyType>& data, double start_percent, double end_percent) {
    sort(data.begin(), data.end());
    data.erase(unique(data.begin(), data.end()), data.end());

    size_t start_key_ind = static_cast<size_t>(start_percent * data.size());
    size_t end_key_ind = static_cast<size_t>(end_percent * data.size());

    vector<KeyType> temp(data.begin() + start_key_ind, data.begin() + end_key_ind);
    data = std::move(temp);
}

template <typename KeyType>
double computeOverflowRatioNoUpperBound(const vector<KeyType>& data,
                                        const StructSegment<KeyType>& seg) {
    KeyType seg_lower = seg.seg_lower;
    KeyType seg_upper = seg.seg_upper;
    KeyType seg_len = seg_upper - seg_lower;
    int box_num = (int)ceil((double)seg_len / (double)seg.box_range);
    double cumOverflow = 0.0;
    double cumKeys = 0.0;
    auto startIt = lower_bound(data.begin(), data.end(), seg_lower);

    for (int i = 0; i < box_num; i++) {
        KeyType boxUpper = min(seg_lower + (i + 1) * seg.box_range - 1, seg_upper);

        int countBox = countKeysInInterval(data, startIt, boxUpper);
        startIt += countBox;
        cumKeys += countBox;
        if (countBox > BOX_CAPACITY) cumOverflow += (countBox - BOX_CAPACITY);
    }
    return (cumKeys > 0) ? (cumOverflow / cumKeys) : 0.0;
}

// did not account for the fact that some boxes will have 0 keys
vector<Block<KeyType>> computeBlocksByWind(const vector<KeyType>& data, KeyType window_size) {
    vector<Block<KeyType>> blocks;
    size_t start_ind = 0;
    KeyType last = data[data.size() - 1];
    KeyType start_key = data[start_ind];

    while (start_ind < data.size()) {
        Block<KeyType> block;
        block.startKey = start_key;

        // window size reaches past end of data
        if (start_key > KEY_MAX - window_size || start_key + window_size > last) {
            block.endKey = (last == KEY_MAX) ? KEY_MAX : last + 1;
            block.range = block.endKey - block.startKey;
            blocks.push_back(block);
            break;
        }

        block.range = window_size;
        KeyType end_key = start_key + window_size; // exclusive
        block.endKey = end_key;
        blocks.push_back(block);

        start_key = end_key;
        size_t end_ind = upper_bound(data.begin() + start_ind, data.end(), end_key - 1) -
                         data.begin();
        start_ind = end_ind;
    }

    return blocks;
}

// input: where to start (which box percentage-wise), where to end, list of window sizes
// output: csv file with window size, length of segment, ratio of underflow, ratio of overflow
// for each window size: finds the starting box, then forms segments (with the given window size)
// of increasing size until we reach the ending box (which is also a percentage). for each segment,
// calculate the ratio of underflow and overflow.
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

string toScientificNotation(double value) {
    if (value == 0.0) {
        return "0";
    }

    if (value < 1.0 && value > 0.0) {
        ostringstream oss;
        oss << scientific << setprecision(0) << value;
        string result = oss.str();

        size_t e_pos = result.find('e');
        if (e_pos != string::npos) {
            string mantissa = result.substr(0, e_pos);
            string exponent = result.substr(e_pos + 2);

            while (mantissa.back() == '0') {
                mantissa.pop_back();
            }
            if (mantissa.back() == '.') {
                mantissa.pop_back();
            }

            size_t i = 0;
            while (exponent[i] == '0') {
                i++;
            }

            return mantissa + "e-" + exponent.substr(i);
        }
        return result;
    }

    if (value < 1000.0 && value == static_cast<int64_t>(value)) {
        return to_string(static_cast<int64_t>(value));
    }

    if (value >= 1000.0) {
        ostringstream oss;
        oss << scientific << setprecision(0) << value;
        string result = oss.str();

        size_t e_pos = result.find('e');
        if (e_pos != string::npos) {
            string mantissa = result.substr(0, e_pos);
            string exponent = result.substr(e_pos + 2); // remove e+

            while (mantissa.back() == '0') {
                mantissa.pop_back();
            }
            if (mantissa.back() == '.') {
                mantissa.pop_back();
            }

            size_t i = 0;
            while (exponent[i] == '0') {
                i++;
            }

            return mantissa + "e" + exponent.substr(i);
        }
        return result;
    }

    ostringstream oss;
    oss << fixed << setprecision(6) << value;
    string result = oss.str();

    while (result.back() == '0') {
        result.pop_back();
    }
    if (result.back() == '.') {
        result.pop_back();
    }

    return result;
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
