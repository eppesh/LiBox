CXX = g++

SRC = ./test/benchmark.cpp
TARGET = ./test/benchmark

THREAD_MODE ?= 1

COMMON_FLAGS = --std=c++17 -faligned-new -march=native -fopenmp 

all:
	$(CXX) -O3 $(COMMON_FLAGS) $(SRC) -o $(TARGET)

debug:
	$(CXX) -g $(COMMON_FLAGS) $(SRC) -o $(TARGET)_debug

clean:
	rm -f $(TARGET) $(TARGET)_debug