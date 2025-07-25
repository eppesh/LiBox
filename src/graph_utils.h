
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