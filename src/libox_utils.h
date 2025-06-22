#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

static const int BLOCK_SIZE = 64;
static const double BOX_CAPACITY = 64.0;

namespace liboxns {
template <typename KeyType>
struct Block {
    KeyType startKey;
    KeyType endKey;
    KeyType range;
};

template <typename KeyType>
struct StructSegment {
    KeyType startIndex;
    KeyType endIndex;
    KeyType seg_lower;
    KeyType seg_upper;
    KeyType box_range;
};

template <typename KeyType>
vector<Block<KeyType>> computeBlocks(const vector<KeyType>& data,
                                     int blockSize = BLOCK_SIZE,
                                     KeyType newseg_lower = 0,
                                     KeyType newseg_upper = 0) {
    vector<Block<KeyType>> blocks;
    size_t n = data.size();
    size_t start;
    size_t end;
    for (size_t i = 0; i < n; i += blockSize) {
        if (i == 0) {
            start = 0;
        } else {
            start = end;
        }
        end = min(n, i + blockSize);
        Block<KeyType> b;
        if (i == 0) {
            b.startKey = (newseg_lower == 0 && data[start] == 0)
                             ? 0
                             : (newseg_lower == 0 ? data[start] - 1 : newseg_lower);
        } else {
            b.startKey = data[start];
        }
        if (end == n) {
            // b.endKey = newseg_upper == 0 ? data[end] + 1 : newseg_upper;
            if (newseg_upper == 0) {
                b.endKey = (data[end - 1] == UINT64_MAX) ? UINT64_MAX : data[end - 1] + 1;
            } else {
                b.endKey = newseg_upper;
            }
        } else {
            b.endKey = data[end];
        }
        b.range = b.endKey - b.startKey;
        blocks.push_back(b);
    }
    return blocks;
}

template <typename KeyType>
StructSegment<KeyType>
createSegment(const vector<Block<KeyType>>& blocks, KeyType i, KeyType j, KeyType box_range = 0) {
    StructSegment<KeyType> seg;
    seg.startIndex = i;
    seg.endIndex = j;
    seg.seg_lower = blocks[i].startKey;
    seg.seg_upper = blocks[j].endKey;
    if (box_range) {
        seg.box_range = box_range;
    } else {
        KeyType minRange = numeric_limits<KeyType>::max();
        for (int idx = i; idx <= j; idx++) {
            minRange = min(minRange, blocks[idx].range);
        }
        seg.box_range = minRange;
    }
    return seg;
}

template <typename KeyType>
int countKeysInInterval(const vector<KeyType>& data,
                        typename vector<KeyType>::const_iterator startIt,
                        KeyType U) {
    auto endIt = upper_bound(startIt, data.end(), U);
    return endIt - startIt;
}

template <typename KeyType>
double computeUnderflowRatioAccurate(const vector<KeyType>& data,
                                     const StructSegment<KeyType>& seg) {
    KeyType seg_lower = seg.seg_lower;
    KeyType seg_upper = seg.seg_upper;
    KeyType seg_len = seg_upper - seg_lower;
    int box_num = (int)ceil((double)seg_len / (double)seg.box_range);
    double cumUnderflow = 0.0;
    double cumKeys = 0.0;
    auto startIt = lower_bound(data.begin(), data.end(), seg_lower);

    for (int i = 0; i < box_num; i++) {
        KeyType boxUpper = min(seg_lower + (i + 1) * seg.box_range - 1, seg_upper);

        int countBox = countKeysInInterval(data, startIt, boxUpper);
        startIt += countBox;
        cumKeys += countBox;
        if (countBox < BOX_CAPACITY) cumUnderflow += (BOX_CAPACITY - countBox);
    }
    return (cumKeys > 0) ? (cumUnderflow / cumKeys) : 0.0;
}

template <typename KeyType>
double computeOverflowRatioAccurate(const vector<KeyType>& data,
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

        if (countBox >= 128) {
            return std::numeric_limits<int>::max();
        }

        startIt += countBox;
        cumKeys += countBox;
        if (countBox > BOX_CAPACITY) cumOverflow += (countBox - BOX_CAPACITY);
    }
    return (cumKeys > 0) ? (cumOverflow / cumKeys) : 0.0;
}

template <typename KeyType>
StructSegment<KeyType> mergeCandidate(const vector<StructSegment<KeyType>>& initSegments,
                                      const vector<Block<KeyType>>& blocks,
                                      KeyType startIdx,
                                      KeyType endIdx,
                                      KeyType box_range) {
    KeyType globalStart = initSegments[startIdx].startIndex;
    KeyType globalEnd = initSegments[endIdx].endIndex;
    return createSegment(blocks, globalStart, globalEnd, box_range);
}

template <typename KeyType>
vector<StructSegment<KeyType>> partitionSegmentsOverall(const vector<Block<KeyType>>& blocks,
                                                        const vector<KeyType>& data,
                                                        double underflowThreshold,
                                                        int maxMergeCount) {
    vector<StructSegment<KeyType>> segments;
    int m = blocks.size();
    if (m == 0) return segments;
    KeyType i = 0;
    while (i < m) {
        int mergeCount = 0;
        KeyType j = i + 1;
        while (j < m && mergeCount < maxMergeCount) {
            StructSegment<KeyType> candidate = createSegment(blocks, i, j);
            double uf = computeUnderflowRatioAccurate(data, candidate);
            if (uf <= underflowThreshold) {
                mergeCount = 0;
            } else {
                mergeCount++;
            }
            j++;
        }
        segments.push_back(createSegment(blocks, i, j - mergeCount - 1));
        i = j - mergeCount;
    }
    return segments;
}

template <typename KeyType>
vector<StructSegment<KeyType>> expandSegments(const vector<StructSegment<KeyType>>& initSegments,
                                              const vector<Block<KeyType>>& blocks,
                                              const vector<KeyType>& data,
                                              double underflowThreshold,
                                              double overflowThreshold,
                                              int maxMergeCount) {
    vector<StructSegment<KeyType>> merged;
    int n = initSegments.size();
    KeyType i = 0;
    while (i < n) {
        StructSegment<KeyType> current = initSegments[i];
        KeyType next = i + 1;
        int mergeCount = 0;
        while (next < n && mergeCount < maxMergeCount) {
            StructSegment<KeyType> candidate =
                mergeCandidate(initSegments, blocks, i, next, current.box_range);
            double uf = computeUnderflowRatioAccurate(data, candidate);
            double of = computeOverflowRatioAccurate(data, candidate);
            if (uf <= underflowThreshold && of <= overflowThreshold) {
                current = candidate;
                mergeCount = 0;
            } else {
                mergeCount++;
            }
            next++;
        }
        merged.push_back(current);
        i = next - mergeCount;
    }
    return merged;
}
} // namespace liboxns