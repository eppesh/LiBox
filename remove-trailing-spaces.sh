#!/bin/bash

for f in src/libox.h src/partition_optimization.cpp test/benchmark.cpp
do
    echo $f
    sed -i 's/[ \t]*$//' $f
done
