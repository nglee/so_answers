all:
	g++  -std=c++11 -O2 -o cpu lsearch.cpp
	nvcc -std=c++11 -O2 -o gpu lsearch.cu -arch=sm_53
