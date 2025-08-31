#!/bin/bash

make
make partition

minN=500
maxN=500
stepN=500

for ((max_box_in_seg=minN; max_box_in_seg<=maxN; max_box_in_seg+=stepN)) do
    seg_file="100M_segments-${max_box_in_seg}.csv"
    if ! [ -f $seg_file ]; then
        ./partition_optimization 100m.csv $seg_file
    fi
    echo "Running benchmark for max_box_in_seg: $max_box_in_seg"
    output=$(numactl -N -0 -m 0 ./test/benchmark --keys_file=longitudes.csv \
    --keys_file_type=text --config_file_path=$seg_file \
    --init_num_keys=100000000 --total_num_keys=200000000 \
    --batch_size=200000000 --insert_frac=0.5 --thread_num=84 \
    --print_batch_stats 2>&1)
    
    # Extract the first two numbers from batch throughput line
    throughput_numbers=$(echo "$output" | grep "batch throughput:" | head -1 | sed 's/.*batch throughput:[[:space:]]*\([0-9.]*\) Mop\/s (lookups),[[:space:]]*\([0-9.]*\) Mop\/s.*/\1 \2/')
    echo "max_box_in_seg: $max_box_in_seg, throughput: $throughput_numbers"
done