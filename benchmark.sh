#!/bin/bash

cpu=$(sudo lshw -C CPU | grep 'product') && cpu=${cpu#*: } && cpu=${cpu// /_} && cpu=$(sed 's/[)(]//g' <<< $cpu)
filename="${cpu}.json"

mkdir -p build
cd build || exit
rm -rf ./*

cmake -E env CXX=clang++ CC=clang cmake -GNinja ..
ninja test_driver benchmark_driver
test_driver/test_driver --gtest_brief=1
benchmark_driver/benchmark_driver -dev --benchmark_format=json --benchmark_out="$filename" --benchmark_context=CPU="$cpu" --benchmark_repetitions=10

cp "$filename" "../../Thesis/data/"
