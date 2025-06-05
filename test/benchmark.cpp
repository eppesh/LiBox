#include <immintrin.h>
#include <omp.h>
#include <xmmintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <cstring> 
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../src/libox.h"

using namespace std;

using FlagMap = unordered_map<string, string>;

FlagMap parse_flags (int argc, char* argv[]) {
    FlagMap flags;
    for (int i = 1; i < argc; i++) {
        string arg (argv[i]);
        if (arg.substr (0, 2) == "--") {
            arg = arg.substr (2);
            auto pos = arg.find ('=');
            if (pos != string::npos) {
                string key = arg.substr (0, pos);
                string value = arg.substr (pos + 1);
                flags[key] = value;
            } else {
                flags[arg] = "true";
            }
        }
    }
    return flags;
}

string get_required (const FlagMap& flags, const string& key) {
    auto it = flags.find (key);
    if (it == flags.end ()) {
        throw runtime_error ("Missing required flag: " + key);
    }
    return it->second;
}

string get_with_default (const FlagMap& flags, const string& key, const string& default_value) {
    auto it = flags.find (key);
    return (it == flags.end ()) ? default_value : it->second;
}

bool get_boolean_flag (const FlagMap& flags, const string& key) {
    auto it = flags.find (key);
    if (it != flags.end ()) {
        string val = it->second;
        return (val == "true" || val == "1");
    }
    return false;
}

//==============================================
// Main process: Read data, perform partition optimization, build index,
// execute operations, and perform local dynamic adjustments
//==============================================
using KeyType = int64_t;
using ValueType = int64_t; 

vector<pair<KeyType, KeyType>> read_range_queries(const string& range_file_path) {
    vector<pair<KeyType, KeyType>> range_queries;
    ifstream fin(range_file_path);
    if (!fin) {
        throw runtime_error("Unable to open range queries file: " + range_file_path);
    }
    
    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        
        size_t comma_pos = line.find(',');
        if (comma_pos == string::npos) {
            cerr << "Warning: Invalid line format in range queries file: " << line << endl;
            continue;
        }
        
        try {
            KeyType start_key = stoll(line.substr(0, comma_pos));
            KeyType end_key = stoll(line.substr(comma_pos + 1));
            
            if (start_key <= end_key) {
                range_queries.push_back({start_key, end_key});
            } else {
                cerr << "Warning: Invalid range [" << start_key << ", " << end_key << "], start > end" << endl;
            }
        } catch (const exception& e) {
            cerr << "Warning: Error parsing line: " << line << " - " << e.what() << endl;
        }
    }
    
    fin.close();
    cout << "Read " << range_queries.size() << " range queries from file." << endl;
    return range_queries;
}

bool validate_range_results(const vector<pair<KeyType, ValueType>>& results, 
                           KeyType start_key, KeyType end_key) {
    for (const auto& result : results) {
        if (result.first < start_key || result.first > end_key) {
            return false;
        }
    }
    
    return true;
}

vector<pair<KeyType, int>> read_scan_queries(const string& scan_file_path) {
    vector<pair<KeyType, int>> scan_queries;
    ifstream fin(scan_file_path);
    if (!fin) {
        throw runtime_error("Unable to open scan queries file: " + scan_file_path);
    }
    
    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        
        size_t comma_pos = line.find(',');
        if (comma_pos == string::npos) {
            cerr << "Warning: Invalid line format in scan queries file: " << line << endl;
            continue;
        }
        
        try {
            KeyType start_key = stoll(line.substr(0, comma_pos));
            int scan_count = stoi(line.substr(comma_pos + 1));
            
            if (scan_count > 0) {
                scan_queries.push_back({start_key, scan_count});
            } else {
                cerr << "Warning: Invalid scan count " << scan_count << " for key " << start_key << endl;
            }
        } catch (const exception& e) {
            cerr << "Warning: Error parsing scan query line: " << line << " - " << e.what() << endl;
        }
    }
    
    fin.close();
    cout << "Read " << scan_queries.size() << " scan queries from file." << endl;
    return scan_queries;
}

bool validate_scan_results(const vector<pair<KeyType, ValueType>>& results, 
                          KeyType start_key, int expected_count) {
    if (results.size() > expected_count) {
        return false;
    }
    
    // for (size_t i = 1; i < results.size(); i++) {
    //     if (results[i].first <= results[i-1].first) {
    //         return false;
    //     }
    // }
    
    for (const auto& result : results) {
        if (result.first < start_key) {
            return false;
        }
    }
    
    return true;
}

int main (int argc, char* argv[]) {
    // Parse command-line arguments
    auto flags = parse_flags (argc, argv);
    string keys_file_path = get_required (flags, "keys_file");
    string keys_file_type = get_required (flags, "keys_file_type");
    string config_file_path = get_required (flags, "config_file_path");
    int init_num_keys = stoi (get_required (flags, "init_num_keys"));
    int total_num_keys = stoi (get_required (flags, "total_num_keys"));
    int thread_num = stod (get_with_default (flags, "thread_num", "1"));
    double insert_frac = stod (get_with_default (flags, "insert_frac", "0.5"));
    int batch_size = stoi (get_with_default (flags, "batch_size", "100000"));
    bool print_batch_stats = get_boolean_flag (flags, "print_batch_stats");
    std::string output_path = get_with_default (flags, "output_path", "./out.csv");

    // Range search
    string range_queries_file = get_with_default (flags, "range_queries_file", "");
    bool enable_range_search = get_boolean_flag (flags, "enable_range_search");
    bool validate_range_results_flag = get_boolean_flag (flags, "validate_range_results");
    int range_search_repeat = stoi (get_with_default (flags, "range_search_repeat", "1"));

    // scan 
    bool enable_scan_mode = get_boolean_flag(flags, "enable_scan_mode");
    string scan_queries_file = get_with_default(flags, "scan_queries_file", "");
    bool validate_scan_results_flag = get_boolean_flag(flags, "validate_scan_results");
    int scan_repeat = stoi(get_with_default(flags, "scan_repeat", "1"));
    double scan_frac = stod(get_with_default(flags, "scan_frac", "0.95"));

    // Initial partition optimization thresholds
    double init_underflow = stod (get_with_default (flags, "init_underflow", "0.5"));
    double init_overflow = stod (get_with_default (flags, "init_overflow", "0.1"));

    cout << fixed << setprecision (3);
    cout << "Parameters:\n"
         << "  keys_file: " << keys_file_path << "\n"
         << "  keys_file_type: " << keys_file_type << "\n"
         << "  init_num_keys: " << init_num_keys << "\n"
         << "  total_num_keys: " << total_num_keys << "\n"
         << "  init_underflow: " << init_underflow << "\n"
         << "  init_overflow: " << init_overflow << "\n"
         << "  thread_num: " << thread_num << "\n"
         << "  insert_frac:" << insert_frac << "\n"
         << "  batch_size:" << batch_size << "\n"
         << "  enable_range_search:" << enable_range_search << "\n"
         << "  range_queries_file:" << range_queries_file << "\n"
         << "  range_search_repeat:" << range_search_repeat << "\n"
         << "  validate_range_results:" << validate_range_results_flag << "\n";

    cout << "  enable_scan_mode: " << enable_scan_mode << "\n"
         << "  scan_queries_file: " << scan_queries_file << "\n"
         << "  scan_frac: " << scan_frac << "\n"
         << "  scan_repeat: " << scan_repeat << "\n"
         << "  validate_scan_results: " << validate_scan_results_flag << "\n";

    vector<pair<KeyType, KeyType>> range_queries;
    if (enable_range_search && !range_queries_file.empty()) {
        range_queries = read_range_queries(range_queries_file);
        if (range_queries.empty()) {
            cout << "Warning: No valid range queries found, disabling range search." << endl;
            enable_range_search = false;
        }
    }

    vector<int64_t> file_keys;
    if (keys_file_type == "binary") {
        ifstream fin (keys_file_path, ios::binary);
        if (!fin) {
            throw runtime_error ("Unable to open file: " + keys_file_path);
        }
        int64_t num;
        int count = 0;
        while (fin.read (reinterpret_cast<char*> (&num), sizeof (num)) && count < total_num_keys) {
            file_keys.push_back (num);
            count++;
        }
        fin.close ();
    } else if (keys_file_type == "text") {
        ifstream fin (keys_file_path);
        if (!fin) {
            throw runtime_error ("Unable to open file: " + keys_file_path);
        }
        string line;
        int count = 0;
        while (getline (fin, line) && count < total_num_keys) {
            if (line.empty ()) continue;
            file_keys.push_back (stoll (line));
            count++;
        }
        fin.close ();
    } else {
        throw runtime_error ("Unknown file type: " + keys_file_type);
    }
    cout << "Read " << file_keys.size () << " numbers from file.\n";

    TaskQueue splitTaskQueue;
    vector<int> taskCoreIDs = {1,3,5,7};
    vector<thread> taskPool;
    for (size_t i = 0; i < taskCoreIDs.size(); i++) {
        taskPool.emplace_back([&splitTaskQueue]() {
            std::packaged_task<void()> task;
            while (splitTaskQueue.popTask(task)) {
                task();
            }
        });
        setThreadAffinity(taskPool.back(), taskCoreIDs[i]);
    }

    liboxns::LiBox<KeyType, ValueType> index (init_underflow, init_overflow, &splitTaskQueue, thread_num);
    index.loadConfigByFile (config_file_path);
    vector<KeyType> init_keys (file_keys.begin (), file_keys.begin () + init_num_keys);
    index.buildIndex (&init_keys);

    int i = init_num_keys;
    long long cumulative_inserts = 0;
    long long cumulative_lookups = 0;
    long long cumulative_found = 0;
    int num_inserts_per_batch = static_cast<int> (batch_size * insert_frac);
    int num_lookups_per_batch = batch_size - num_inserts_per_batch;
    double cumulative_insert_time = 0;
    double cumulative_lookup_time = 0;
    double cumulative_total_mops;

    int batch_no = 0;
    while (true) {
        batch_no++;

        double batch_lookup_time = 0.0;
        double batch_insert_time = 0.0;

        if (num_lookups_per_batch > 0) {
            vector<KeyType> lookup_keys;
            random_device rd;
            mt19937 gen (rd ());
            uniform_int_distribution<> dis (0, init_num_keys - 1);

            for (int j = 0; j < num_lookups_per_batch; j++) {
                lookup_keys.push_back (file_keys[dis (gen)]);
            }

            int batch_found = 0;
            chrono::high_resolution_clock::time_point lookups_start_time, lookups_end_time;
            lookups_start_time = chrono::high_resolution_clock::now ();
            omp_set_num_threads(thread_num);
            #pragma omp parallel for reduction(+:batch_found)
            for (int j = 0; j < num_lookups_per_batch; j++) {
                if (index.searchKey(lookup_keys[j]).value == 1) {
                    batch_found++;
                }
            }
            lookups_end_time = chrono::high_resolution_clock::now ();

            batch_lookup_time =
                chrono::duration_cast<chrono::nanoseconds> (lookups_end_time - lookups_start_time).count ();
            cumulative_lookup_time += batch_lookup_time;
            cumulative_lookups += num_lookups_per_batch;
            cumulative_found += batch_found;
        }

        int num_actual_inserts = min (num_inserts_per_batch, total_num_keys - i);
        int num_keys_after_batch = i + num_actual_inserts;
        if (num_inserts_per_batch > 0) {
            int inserted = 0;

            chrono::high_resolution_clock::time_point inserts_start_time, inserts_end_time;
            inserts_start_time = chrono::high_resolution_clock::now ();
            omp_set_num_threads(thread_num);
            #pragma omp parallel for reduction(+:inserted)
            for (int j = i; j < num_keys_after_batch; j++) {
                if (index.insertKeyValue(file_keys[j], 1).status == liboxns::InsertStatus::SUCCESS) {
                    inserted++;
                }
            }
            
            index.waitForCurrentBatch();
            inserts_end_time = chrono::high_resolution_clock::now ();

            batch_insert_time =
                chrono::duration_cast<chrono::nanoseconds> (inserts_end_time - inserts_start_time).count ();
            cumulative_insert_time += batch_insert_time;
            cumulative_inserts += inserted;
            i = num_keys_after_batch;
        }

        if (print_batch_stats) {
            int num_batch_operations = num_lookups_per_batch + num_actual_inserts;
            double batch_time = batch_lookup_time + batch_insert_time;
            long long cumulative_operations = cumulative_lookups + cumulative_inserts;
            double cumulative_time = cumulative_lookup_time + cumulative_insert_time;

            double batch_lookup_mops = num_lookups_per_batch / batch_lookup_time * 1000;
            double batch_insert_mops = num_actual_inserts / batch_insert_time * 1000;
            double batch_total_mops = num_batch_operations / batch_time * 1000;

            double cumulative_lookup_mops = cumulative_lookups / cumulative_lookup_time * 1000;
            double cumulative_insert_mops = cumulative_inserts / cumulative_insert_time * 1000;
            cumulative_total_mops = cumulative_operations / cumulative_time * 1000;

            cout << "Batch " << batch_no << ", cumulative ops: " << cumulative_operations << "\n\tbatch throughput:\t"
                 << batch_lookup_mops << " Mop/s (lookups),\t" << batch_insert_mops << " Mop/s (inserts),\t"
                 << batch_total_mops << " Mop/s (total)"
                 << "\n\tcumulative throughput:\t" << cumulative_lookup_mops << " Mop/s (lookups),\t"
                 << cumulative_insert_mops << " Mop/s (inserts),\t" << cumulative_total_mops << " Mop/s (total)"
                 << endl;
        }

        if (insert_frac == 0) {
            if (cumulative_lookups >= total_num_keys - init_num_keys) {
                break;
            }
        } else {
            if (i >= total_num_keys) {
                break;
            }
        }
    }

    if (enable_range_search && !range_queries.empty()) {
        cout << "\n=== Range Search Performance Test ===" << endl;
        cout << "Testing " << range_queries.size() << " range queries, " 
             << range_search_repeat << " times each..." << endl;

        vector<vector<pair<KeyType, ValueType>>> all_results(range_queries.size());
        long long total_range_operations = range_queries.size() * range_search_repeat;
        long long total_results_found = 0;
        
        chrono::high_resolution_clock::time_point range_start_time, range_end_time;
        range_start_time = chrono::high_resolution_clock::now();
        
        omp_set_num_threads(thread_num);
        #pragma omp parallel for reduction(+:total_results_found)
        for (int repeat = 0; repeat < range_search_repeat; repeat++) {
            for (size_t q = 0; q < range_queries.size(); q++) {
                auto result = index.rangeSearch(range_queries[q].first, range_queries[q].second);
                
                if (repeat == 0) {
                    all_results[q] = result.results;
                }
                
                total_results_found += result.results.size();
            }
        }
        
        range_end_time = chrono::high_resolution_clock::now();
        
        double total_range_time = chrono::duration_cast<chrono::nanoseconds>(range_end_time - range_start_time).count();
        double range_search_mops = total_range_operations / total_range_time * 1000;
        double avg_results_per_query = (double)total_results_found / total_range_operations;
        double avg_time_per_query_us = total_range_time / total_range_operations / 1000.0;
        
        cout << "Range Search Results:" << endl;
        cout << "  Total queries executed: " << total_range_operations << endl;
        cout << "  Total results found: " << total_results_found << endl;
        cout << "  Average results per query: " << fixed << setprecision(2) << avg_results_per_query << endl;
        cout << "  Throughput: " << fixed << setprecision(3) << range_search_mops << " Mop/s" << endl;
        cout << "  Average time per query: " << fixed << setprecision(3) << avg_time_per_query_us << " μs" << endl;
        
        if (validate_range_results_flag) {
            cout << "\n=== Validating Range Search Results ===" << endl;
            
            int validation_errors = 0;
            long long validation_total_results = 0;
            
            for (size_t q = 0; q < range_queries.size(); q++) {
                KeyType start_key = range_queries[q].first;
                KeyType end_key = range_queries[q].second;
                const auto& results = all_results[q];
                
                validation_total_results += results.size();
                
                if (!validate_range_results(results, start_key, end_key)) {
                    validation_errors++;
                    cout << "  ERROR in query " << q << " [" << start_key << ", " << end_key << "]: ";
                    
                    bool has_out_of_range = false;
                    
                    for (const auto& result : results) {
                        if (result.first < start_key || result.first > end_key) {
                            has_out_of_range = true;
                            break;
                        }
                    }
                    
                    if (has_out_of_range) cout << "Keys out of range ";
                    cout << endl;
                    
                    if (validation_errors <= 5) {
                        cout << "    First few results: ";
                        for (size_t i = 0; i < min(size_t(5), results.size()); i++) {
                            cout << "(" << results[i].first << "," << results[i].second << ") ";
                        }
                        cout << endl;
                    }
                }
            }
            
            if (validation_errors == 0) {
                cout << "  ✓ All " << range_queries.size() << " range queries passed validation!" << endl;
                cout << "  ✓ Total " << validation_total_results << " results are all within correct ranges." << endl;
            } else {
                cout << "  ✗ " << validation_errors << " out of " << range_queries.size() 
                     << " queries failed validation!" << endl;
            }
        }
        
        cout << "\nSample query results:" << endl;
        for (size_t q = 0; q < min(size_t(5), range_queries.size()); q++) {
            const auto& results = all_results[q];
            cout << "  Query " << q << " [" << range_queries[q].first << ", " << range_queries[q].second 
                 << "]: " << results.size() << " results";
            if (!results.empty()) {
                cout << " (first: " << results[0].first << ", last: " << results.back().first << ")";
            }
            cout << endl;
        }
    }

    vector<pair<KeyType, int>> scan_queries;
    if (enable_scan_mode && !scan_queries_file.empty()) {
        scan_queries = read_scan_queries(scan_queries_file);
        if (scan_queries.empty()) {
            cout << "Warning: No valid scan queries found, disabling scan mode." << endl;
            enable_scan_mode = false;
        }
    }

    if (enable_scan_mode) {
        num_inserts_per_batch = static_cast<int>(batch_size * (1.0 - scan_frac));
        int num_scans_per_batch = batch_size - num_inserts_per_batch;
        
        long long cumulative_scans = 0;
        long long cumulative_scan_results = 0;
        double cumulative_scan_time = 0;
        
        cout << "Running Optimized Scan Mode: " << (scan_frac * 100) << "% scan, " 
            << ((1.0 - scan_frac) * 100) << "% insert per batch" << endl;
            
        batch_no = 0;
        
        while (true) {
            batch_no++;
            
            double batch_scan_time = 0.0;
            double batch_insert_time = 0.0;
            long long batch_scan_results = 0;
            
            if (num_scans_per_batch > 0 && !scan_queries.empty()) {
                vector<pair<KeyType, int>> batch_scan_queries;
                batch_scan_queries.reserve(num_scans_per_batch);
                
                random_device rd;
                mt19937 gen(rd());
                uniform_int_distribution<> dis(0, scan_queries.size() - 1);
                
                for (int j = 0; j < num_scans_per_batch; j++) {
                    int query_idx = dis(gen);
                    batch_scan_queries.push_back(scan_queries[query_idx]);
                }
                
                sort(batch_scan_queries.begin(), batch_scan_queries.end(),
                    [](const pair<KeyType, int>& a, const pair<KeyType, int>& b) {
                        return a.first < b.first;
                    });
                
                chrono::high_resolution_clock::time_point scan_start_time, scan_end_time;
                scan_start_time = chrono::high_resolution_clock::now();
                
                const int MAX_SCAN_RESULTS = 1000;
                
                omp_set_num_threads(thread_num);
                #pragma omp parallel reduction(+:batch_scan_results)
                {
                    pair<KeyType, ValueType> thread_scan_buffer[MAX_SCAN_RESULTS];
                    
                    #pragma omp for
                    for (int j = 0; j < num_scans_per_batch; j++) {
                        KeyType start_key = batch_scan_queries[j].first;
                        int scan_count = min(batch_scan_queries[j].second, MAX_SCAN_RESULTS);
                        
                        size_t results_count = index.scan_optimized(start_key, scan_count, thread_scan_buffer);
                        batch_scan_results += results_count;
                    }
                }
                
                scan_end_time = chrono::high_resolution_clock::now();
                batch_scan_time = chrono::duration_cast<chrono::nanoseconds>(scan_end_time - scan_start_time).count();
                cumulative_scan_time += batch_scan_time;
                cumulative_scans += num_scans_per_batch;
                cumulative_scan_results += batch_scan_results;
            }
            
            int num_actual_inserts = 0;
            if (num_inserts_per_batch > 0 && i < total_num_keys) {
                num_actual_inserts = min(num_inserts_per_batch, total_num_keys - i);
                int num_keys_after_batch = i + num_actual_inserts;
                
                int inserted = 0;
                
                chrono::high_resolution_clock::time_point inserts_start_time, inserts_end_time;
                inserts_start_time = chrono::high_resolution_clock::now();
                omp_set_num_threads(thread_num);
                #pragma omp parallel for reduction(+:inserted)
                for (int j = i; j < num_keys_after_batch; j++) {
                    if (index.insertKeyValue(file_keys[j], 1).status == liboxns::InsertStatus::SUCCESS) {
                        inserted++;
                    }
                }
                
                index.waitForCurrentBatch();
                inserts_end_time = chrono::high_resolution_clock::now();
                
                batch_insert_time = chrono::duration_cast<chrono::nanoseconds>(inserts_end_time - inserts_start_time).count();
                cumulative_insert_time += batch_insert_time;
                cumulative_inserts += inserted;
                i = num_keys_after_batch;
            }
            
            if (print_batch_stats) {
                int num_batch_operations = num_scans_per_batch + num_actual_inserts;
                double batch_time = batch_scan_time + batch_insert_time;
                long long cumulative_operations = cumulative_scans + cumulative_inserts;
                double cumulative_time = cumulative_scan_time + cumulative_insert_time;
                
                double batch_scan_mops = (batch_scan_time > 0) ? num_scans_per_batch / batch_scan_time * 1000 : 0;
                double batch_insert_mops = (batch_insert_time > 0) ? num_actual_inserts / batch_insert_time * 1000 : 0;
                double batch_total_mops = (batch_time > 0) ? num_batch_operations / batch_time * 1000 : 0;
                
                double cumulative_scan_mops = (cumulative_scan_time > 0) ? cumulative_scans / cumulative_scan_time * 1000 : 0;
                double cumulative_insert_mops = (cumulative_insert_time > 0) ? cumulative_inserts / cumulative_insert_time * 1000 : 0;
                double cumulative_total_mops = (cumulative_time > 0) ? cumulative_operations / cumulative_time * 1000 : 0;
                
                double avg_results_per_scan = (cumulative_scans > 0) ? (double)cumulative_scan_results / cumulative_scans : 0;
                
                cout << "Optimized Batch " << batch_no << ", ops: " << cumulative_operations 
                    << " (scans: " << cumulative_scans << ", inserts: " << cumulative_inserts << ")"
                    << "\n\tbatch throughput:\t" << fixed << setprecision(3) << batch_scan_mops << " Mop/s (scans, " 
                    << (num_scans_per_batch > 0 ? (double)batch_scan_results / num_scans_per_batch : 0) << " avg results),\t"
                    << batch_insert_mops << " Mop/s (inserts),\t" << batch_total_mops << " Mop/s (total)"
                    << "\n\tcumulative throughput:\t" << cumulative_scan_mops << " Mop/s (scans, " 
                    << fixed << setprecision(1) << avg_results_per_scan << " avg results),\t"
                    << cumulative_insert_mops << " Mop/s (inserts),\t" << cumulative_total_mops << " Mop/s (total)" << endl;
            }
            
            if (batch_no >= 10) {
                break;
            }
        }
        
        if (validate_scan_results_flag && !scan_queries.empty()) {
            cout << "\n=== Validating Scan Results ===" << endl;
            cout << "Running validation on sample scan queries..." << endl;
            
            int validation_samples = min(10, (int)scan_queries.size());
            int validation_errors = 0;
            
            for (int i = 0; i < validation_samples; i++) {
                KeyType start_key = scan_queries[i].first;
                int scan_count = min(scan_queries[i].second, 100);
                
                vector<pair<KeyType, ValueType>> validation_buffer(scan_count);
                size_t results_count = index.scan_optimized(start_key, scan_count, validation_buffer.data());
                validation_buffer.resize(results_count);
                
                if (!validate_scan_results(validation_buffer, start_key, scan_count)) {
                    validation_errors++;
                    cout << "  ERROR in scan query " << i << " (start_key=" << start_key 
                        << ", count=" << scan_count << ", actual_results=" << results_count << ")" << endl;
                }
            }
            
            if (validation_errors == 0) {
                cout << "  ✓ All " << validation_samples << " validation scans passed!" << endl;
            } else {
                cout << "  ✗ " << validation_errors << " out of " << validation_samples 
                    << " validation scans failed!" << endl;
            }
        }
        
        long long cumulative_operations = cumulative_lookups + cumulative_inserts;
        double cumulative_time = cumulative_lookup_time + cumulative_insert_time;

        double cumulative_lookup_mops = cumulative_lookups / cumulative_lookup_time * 1000;
        double cumulative_insert_mops = cumulative_inserts / cumulative_insert_time * 1000;
        double cumulative_total_mops = cumulative_operations / cumulative_time * 1000;

        cout << "\nCumulative stats: " << batch_no << " batches, " << cumulative_operations << " ops (" << cumulative_lookups
            << " lookups, " << cumulative_inserts << " inserts)"
            << "\n\tcumulative throughput:\t" << cumulative_lookup_mops << " Mop/s (lookups),\t" << cumulative_insert_mops
            << " Mop/s (inserts),\t" << cumulative_total_mops << " Mop/s (total)" << endl;

    }

    std::ifstream fs (output_path);
    if (!fs.is_open ()) {
        std::ofstream ofile;
        ofile.open (output_path, std::ios::app);
        ofile << "timestamp" << ",";
        ofile << "read_ratio" << "," << "insert_ratio" << ",";
        ofile << "key_path" << ",";
        ofile << "throughput" << ",";
        ofile << "init_num_keys" << ",";
        ofile << "thread_num" << ",";
        ofile << "batch_size" << ",";
        ofile << "total_num_keys" << std::endl;
    } else {
        fs.close ();
    }
    std::time_t t = std::time (nullptr);
    char time_str[100];
    std::ofstream ofile;
    ofile.open (output_path, std::ios::app);
    if (std::strftime (time_str, sizeof (time_str), "%Y%m%d%H%M%S", std::localtime (&t))) {
        ofile << time_str << ",";
    }
    ofile << (1 - insert_frac) << "," << insert_frac << ",";
    ofile << keys_file_path << ",";
    ofile << cumulative_total_mops << ",";
    ofile << init_num_keys << ",";
    ofile << thread_num << ",";
    ofile << batch_size << ",";
    ofile << total_num_keys << std::endl;
    ofile.close ();

    if (cumulative_lookups > 0) {
        cout << "Found " << cumulative_found << " matches out of " << cumulative_lookups << " lookups ("
            << (static_cast<double> (cumulative_found) / cumulative_lookups * 100.0) << "%)" << endl;
    }

    splitTaskQueue.shutdown ();
    for (auto& thr : taskPool) {
        if (thr.joinable ()) thr.join ();
    }

    return 0;
}