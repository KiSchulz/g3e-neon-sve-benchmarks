add_executable(benchmark_driver main.cpp)
target_link_libraries(benchmark_driver benchmark::benchmark libpfm.so neon_kernels sve_kernels)