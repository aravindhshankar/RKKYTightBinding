#!/bin/bash

set -xe

#g++ -std=c++17 -I ./ -Wall -Wextra -o test_eigs test_eigs.cpp 
g++ -std=c++17 -I ./ -Wall -Wextra -O3 -o main main.cpp include/*.cpp 
