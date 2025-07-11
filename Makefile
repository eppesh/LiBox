CXX = g++

SRC = ./test/benchmark.cpp
TARGET = ./test/benchmark

THREAD_MODE ?= 1

COMMON_FLAGS = --std=c++17 -faligned-new -march=native -fopenmp 

all:
	$(CXX) -O3 $(COMMON_FLAGS) -DNDEBUG $(SRC) -o $(TARGET)

debug:
	$(CXX) -g $(COMMON_FLAGS) $(SRC) -o $(TARGET)_debug

clean:
	rm -f $(TARGET) $(TARGET)_debug

MAC_FLAGS = -std=c++17 -O3

partition:
	$(CXX) $(MAC_FLAGS) src/partition_optimization.cpp -o partition_optimization

graph:
	$(CXX) $(MAC_FLAGS) src/ratio_by_win.cpp -o ratio_by_win
	$(CXX) $(MAC_FLAGS) src/seg_len_by_win.cpp -o seg_len_by_win

segment:
	$(CXX) $(MAC_FLAGS) src/segmentation.cpp -o segmentation