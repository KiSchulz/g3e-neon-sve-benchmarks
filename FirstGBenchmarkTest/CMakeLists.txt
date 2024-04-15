add_executable(FirstGBenchmarkTest main.cpp)
target_link_libraries(FirstGBenchmarkTest benchmark::benchmark libpfm.so)