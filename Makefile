CXX = g++

SRC = ./test/benchmark.cpp
TARGET = ./test/benchmark

THREAD_MODE ?= 1

COMMON_FLAGS = --std=c++17 -faligned-new -march=native -fopenmp 

all:
	$(CXX) -O3 $(COMMON_FLAGS) -DNDEBUG $(SRC) -o $(TARGET)

debug:
	$(CXX) -g $(COMMON_FLAGS) $(SRC) -o $(TARGET)_debug


MAC_FLAGS ?= $(MAC_RELEASE_FLAGS)

MAC_RELEASE_FLAGS = -std=c++17 -O3 -DNDEBUG
MAC_DEBUG_FLAGS = -std=c++17 -g

partition:
	$(CXX) $(MAC_RELEASE_FLAGS) src/partition_optimization.cpp -o partition_optimization

graph:
	$(CXX) $(MAC_RELEASE_FLAGS) src/ratio_by_win.cpp -o ratio_by_win
	$(CXX) $(MAC_RELEASE_FLAGS) src/seg_len_by_win.cpp -o seg_len_by_win

segment:
	$(CXX) $(MAC_RELEASE_FLAGS) src/segmentation.cpp -o segmentation

partition_debug:
	$(CXX) $(MAC_DEBUG_FLAGS) src/partition_optimization.cpp -o partition_optimization_debug

graph_debug:
	$(CXX) $(MAC_DEBUG_FLAGS) src/ratio_by_win.cpp -o ratio_by_win_debug
	$(CXX) $(MAC_DEBUG_FLAGS) src/seg_len_by_win.cpp -o seg_len_by_win_debug

segment_debug:
	$(CXX) $(MAC_DEBUG_FLAGS) src/segmentation.cpp -o segmentation_debug

clean:
	rm -f $(TARGET) $(TARGET)_debug partition_optimization partition_optimization_debug ratio_by_win ratio_by_win_debug seg_len_by_win seg_len_by_win_debug segmentation segmentation_debug