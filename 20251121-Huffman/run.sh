set -e
set -o xtrace

mkdir -p build
python generate_hist.py > build/hist.bin
clang++ -std=c++17 -O3 -Wall -Wextra -Werror -o build/huffman huffman.cpp
./build/huffman
