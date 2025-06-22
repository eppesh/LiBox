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

partition:
	g++ -std=c++17 -O3 src/partition_optimization.cpp -o partition_optimization

graph:
	g++ -std=c++17 -O3 src/ratio_by_win.cpp -o ratio_by_win