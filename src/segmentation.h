#include <stddef.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "libox_utils.h"
using namespace liboxns;

extern size_t MAX_KEYS_IN_SEGMENT;

namespace seg {
template <typename KeyType>
struct keySegment {
    size_t start_idx;
    KeyType start_key;
    size_t end_idx; // exclusive
    KeyType end_key;
    KeyType window_size;

    size_t cum_keys = 0;
    size_t cum_underflow = 0;
    size_t cum_overflow = 0;

    keySegment(const size_t start_i, KeyType start_k, KeyType window) {
        start_idx = start_i;
        end_idx = start_i;
        start_key = start_k;
        end_key = start_k;
        window_size = window;
    }

    void expand_to(const keySegment<KeyType>& seg) {
        end_idx = seg.end_idx;
        end_key = seg.end_key;
        cum_keys = seg.cum_keys;
        cum_underflow = seg.cum_underflow;
        cum_overflow = seg.cum_overflow;
    }
};
} // namespace seg

using namespace seg;

template <typename KeyType>
size_t find_lower(const std::vector<KeyType>& data,
                  const size_t start_ind,
                  const size_t end_ind,
                  const KeyType key) {
    size_t cur_ind = start_ind;
    auto end = std::min(end_ind, data.size());
    while (cur_ind < end) {
        if (data[cur_ind] >= key) {
            return cur_ind;
        }
        cur_ind++;
    }
    return cur_ind;
}

// Makes the largest segment it can with window_size, starting at start_idx, where the
// over/underflow ratios are under the thresholds. Looks ahead at most max_look_ahead when the
// thresholds are not met in an attempt to meet them.
template <typename KeyType>
keySegment<KeyType> makeSegment(const std::vector<KeyType>& data,
                             const size_t start_idx,
                             const KeyType start_key,
                             const KeyType window_size,
                             const double overflow_threshold,
                             const double underflow_threshold,
                             const size_t max_look_ahead) {
    keySegment<KeyType> seg(start_idx, start_key, window_size);
    keySegment<KeyType> valid_seg(start_idx, start_key, window_size);
#ifndef NDEBUG
    if (data[start_idx] < start_key || (start_idx != 0 && data[start_idx - 1] >= start_key)) {
        std::cout << "uh oh\n"; // debug
    }
#endif

    size_t num_look_ahead = 0;
    while (seg.end_idx < data.size() && num_look_ahead < max_look_ahead) {
        // exclusive
        KeyType new_end_key = std::min(seg.end_key + window_size, data.back() + 1);

        size_t new_end_idx =
            find_lower(data, seg.end_idx, seg.end_idx + BOX_CAPACITY * 2, new_end_key);

#ifndef NDEBUG
        size_t test_new_end_idx =
            std::lower_bound(seg.end_idx + data.begin(), data.end(), new_end_key) - data.begin();

        if (new_end_idx < seg.end_idx + BOX_CAPACITY * 2 && new_end_idx != test_new_end_idx) {
            std::cout << "uhhhhhhhhhh\n";
        }
#endif

        size_t num_keys = new_end_idx - seg.end_idx;
        if (num_keys >= BOX_CAPACITY * 2) {
            break;
        }

        if (seg.cum_keys + num_keys > MAX_KEYS_IN_SEGMENT) {
            break;
        }

        seg.cum_keys += num_keys;
        seg.end_idx = new_end_idx;
        seg.end_key = new_end_key;

        if (num_keys > BOX_CAPACITY) {
            seg.cum_overflow += (num_keys - BOX_CAPACITY);
        } else {
            seg.cum_underflow += (BOX_CAPACITY - num_keys);
        }

        double overflow_ratio = (seg.cum_keys > 0)
                                    ? (static_cast<double>(seg.cum_overflow) / seg.cum_keys)
                                    : 1;
        double underflow_ratio = (seg.cum_keys > 0)
                                     ? (static_cast<double>(seg.cum_underflow) / seg.cum_keys)
                                     : 1;
        if (overflow_ratio > overflow_threshold || underflow_ratio > underflow_threshold) {
            num_look_ahead++;
        } else {
            num_look_ahead = 0;
            valid_seg.expand_to(seg);
        }
    }
    return valid_seg;
}

template <typename KeyType>
keySegment<KeyType> findBestSegment(const std::vector<KeyType>& data,
                                 const size_t start_idx,
                                 const KeyType start_key,
                                 const std::vector<KeyType>& window_candidates,
                                 const double overflow_threshold,
                                 const double underflow_threshold,
                                 const size_t max_look_ahead) {
    keySegment<KeyType> best_seg(start_idx, start_key, 0);
    size_t max_keys = 0;

    for (auto& window_size : window_candidates) {
        keySegment<KeyType> seg = makeSegment(data,
                                           start_idx,
                                           start_key,
                                           window_size,
                                           overflow_threshold,
                                           underflow_threshold,
                                           max_look_ahead);

        if (seg.cum_keys > max_keys) {
            max_keys = seg.cum_keys;
            best_seg = seg;
        }
    }
#ifndef NDEBUG
    if (best_seg.cum_keys < BOX_CAPACITY) {
        std::cout << "not good guys\n";
    }
#endif
    return best_seg;
}

template <typename KeyType>
std::vector<KeyType> getWindowCandidates(const std::vector<KeyType>& data,
                                         const size_t start_idx,
                                         const KeyType start_key) {
    const size_t MAX_NUM_SAMPLES = 20;
    std::vector<KeyType> win_samples;
    size_t cur_idx = start_idx;
    KeyType cur_key = start_key;
    bool breaked_early = false;
    for (int i = 0; i < MAX_NUM_SAMPLES; i++) {
        cur_idx += BOX_CAPACITY;
        if (cur_idx >= data.size()) {
            breaked_early = true;
            break;
        }

        win_samples.push_back(data[cur_idx] - cur_key);
        cur_key = data[cur_idx];
    }
    if (breaked_early) {
        win_samples.push_back(data.back() - cur_key + 1);
    }

    std::sort(win_samples.begin(), win_samples.end());

    const size_t NUM_BETWEEN = 0;
    std::vector<KeyType> win_candidates;
    for (int i = 0; i < win_samples.size() - 1; i++) {
        win_candidates.push_back(win_samples[i]);
        size_t gap = (win_samples[i + 1] - win_samples[i]) / (NUM_BETWEEN + 1);
        for (int j = 1; j <= NUM_BETWEEN; j++) {
            win_candidates.push_back(win_samples[i] + j * gap);
        }
    }
    win_candidates.push_back(win_samples.back());

    return win_candidates;
}

#ifndef NDEBUG
// debug
long long wind_cand_idx_sum = 0;
#endif

template <typename KeyType>
std::vector<StructSegment<KeyType>> toStructSegment(const std::vector<keySegment<KeyType>>& my_segs) {
    std::vector<StructSegment<KeyType>> segs;
    for (const keySegment<KeyType>& my_seg : my_segs) {
        StructSegment<KeyType> seg;
        seg.seg_lower = my_seg.start_key;
        seg.seg_upper = my_seg.end_key;
        seg.box_range = my_seg.window_size;
        segs.push_back(seg);
    }
    return segs;
}

template <typename KeyType>
std::vector<keySegment<KeyType>> calculateSegments(const std::vector<KeyType>& data,
                                                const double overflow_threshold,
                                                const double underflow_threshold,
                                                const size_t max_look_ahead,
                                                const KeyType newseg_lower,
                                                const KeyType newseg_upper) {
    std::vector<keySegment<KeyType>> segments;
    size_t cur_idx = 0;
    KeyType cur_key = newseg_lower <= data[0] ? newseg_lower : data[0];
    while (cur_idx < data.size() && cur_key <= data.back() + 1) {
        if (data.size() - cur_idx < (1 - underflow_threshold) * BOX_CAPACITY) {
            // the remaining data cannot be made into a valid box, so we forcibly create an
            // underflowing segment
            keySegment<KeyType> seg(cur_idx, cur_key, data.back() - cur_key);
            seg.end_idx = data.size();
            seg.end_key = data.back() + 1;
            segments.push_back(seg);
            break;
        }

        std::vector<KeyType> window_candidates = getWindowCandidates(data, cur_idx, cur_key);
        keySegment<KeyType> seg = findBestSegment(data,
                                               cur_idx,
                                               cur_key,
                                               window_candidates,
                                               overflow_threshold,
                                               underflow_threshold,
                                               max_look_ahead);
        cur_idx = seg.end_idx;
        cur_key = seg.end_key;

#ifndef NDEBUG
        // debug
        int win_idx = std::lower_bound(window_candidates.begin(),
                                       window_candidates.end(),
                                       seg.window_size) -
                      window_candidates.begin();
        wind_cand_idx_sum += win_idx;
        // end debug

        if (cur_idx < data.size() && (data[cur_idx] < cur_key || data[cur_idx - 1] >= cur_key)) {
            std::cout << "also oh no"; // debug
        }
#endif

        segments.push_back(seg);
    }

    keySegment<KeyType>& last_seg = segments.back();
    if (newseg_upper > last_seg.end_key) last_seg.end_key = newseg_upper;

    return segments;
}

template <typename KeyType>
bool validateSegments(const std::vector<KeyType>& data,
                      const std::vector<keySegment<KeyType>>& my_segs) {
    std::vector<StructSegment<KeyType>> segs = toStructSegment(my_segs);

    KeyType start_key = data[0];
    KeyType end_key = data[0];
    size_t total_keys = 0;
    for (int i = 0; i < segs.size(); i++) {
        const StructSegment<KeyType>& seg = segs[i];
        if (seg.seg_lower != end_key) {
            return false;
        }
        start_key = end_key;
        end_key = seg.seg_upper;
        if (computeUnderflowRatioAccurate(data, seg) > 0.5 ||
            computeOverflowRatioAccurate(data, seg) > 0.1) {
            return false;
        }

        auto start_it = std::lower_bound(data.begin(), data.end(), start_key);
        int num_keys = countKeysInInterval(data,
                                           start_it,
                                           end_key - 1); // the method expects inclusive
        if (num_keys != my_segs[i].cum_keys) {
            return false;
        }
        total_keys += num_keys;
    }

    if (end_key != data.back() + 1) return false;
    if (total_keys != data.size()) return false;

    return true;
}