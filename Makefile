CXX = g++

SRC = ./test/benchmark.cpp
HEADERS = ./src/libox.h ./src/segmentation.h ./src/libox_utils.h
TARGET = ./test/benchmark

THREAD_MODE ?= 1

# Detect OS and set appropriate flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # Mac-specific flags (no -fopenmp)
    COMMON_FLAGS = --std=c++20 -faligned-new -march=native
else
    # Linux/other OS flags (with -fopenmp)
    COMMON_FLAGS = --std=c++20 -faligned-new -march=native -fopenmp
endif 

all: $(TARGET)

$(TARGET): $(HEADERS) $(SRC)
	$(CXX) -O3 $(COMMON_FLAGS) -DNDEBUG $(SRC) -o $(TARGET)

debug: $(TARGET)_debug

$(TARGET)_debug: $(HEADERS) $(SRC)
	$(CXX) -O0 -g $(COMMON_FLAGS) $(SRC) -o $(TARGET)_debug

prof: $(TARGET)_prof

$(TARGET)_prof: $(HEADERS) $(SRC)
	$(CXX) -O2 -pg $(COMMON_FLAGS) -DNDEBUG $(SRC) -o $(TARGET)_prof


MAC_FLAGS ?= $(MAC_RELEASE_FLAGS)

MAC_RELEASE_FLAGS = -std=c++20 -O3 -DNDEBUG
MAC_DEBUG_FLAGS = -std=c++20 -g

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

count-key:
	$(CXX) $(MAC_RELEASE_FLAGS) -pthread count-key.cpp -o count-key

count-key_debug:
	$(CXX) $(MAC_DEBUG_FLAGS) -pthread count-key.cpp -o count-key_debug

# Gen worst case test targets
gen_worst_case: test/gen_worst_case

test/gen_worst_case: test/gen-worst-case.cpp
	$(CXX) -O3 $(COMMON_FLAGS) -DNDEBUG test/gen-worst-case.cpp -o test/gen_worst_case

gen_worst_case_debug: test/gen_worst_case_debug

test/gen_worst_case_debug: test/gen-worst-case.cpp
	$(CXX) -O0 -g $(COMMON_FLAGS) test/gen-worst-case.cpp -o test/gen_worst_case_debug

clean:
	rm -f $(TARGET) $(TARGET)_debug $(TARGET)_prof partition_optimization partition_optimization_debug ratio_by_win ratio_by_win_debug seg_len_by_win seg_len_by_win_debug segmentation segmentation_debug count-key count-key_debug