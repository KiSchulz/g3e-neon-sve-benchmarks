#ifndef NEON_SVE_BENCH_RANDOM_DATA_GENERATOR_H
#define NEON_SVE_BENCH_RANDOM_DATA_GENERATOR_H

#include <cassert>
#include <memory>
#include <random>

struct RandomDataGenerator {
  std::default_random_engine engine{0};

  std::shared_ptr<char> getRandArr(std::size_t length, const char *prefix = nullptr, std::size_t prefix_len = 0) {
    std::shared_ptr<char> arr{new char[length]};

    if (prefix != nullptr && prefix_len > 0) {
      assert(prefix_len <= length);
      std::copy(prefix, prefix + prefix_len, arr.get());
    }
    initArrWithRand(arr.get() + prefix_len, length - prefix_len);

    return arr;
  }

  void initArrWithRand(void *arr, std::size_t length) {
    std::generate((char *)arr, (char *)arr + length, [this]() {
      std::uniform_int_distribution<char> uni_dist{0, 255};
      return uni_dist(engine);
    });
  }

  void initArrWithRandInRangeD(double *arr, std::size_t length, double low, double high) {
    std::generate(arr, arr + length, [&]() {
      std::uniform_real_distribution<double> uni_dist{low, high};
      return uni_dist(engine);
    });
  }

  void initNBodySystem(double *px, double *py, double *pz, double *vx, double *vy, double *vz, double *mass,
                       std::size_t size) {
    initArrWithRandInRangeD(px, size, -10, 10);
    initArrWithRandInRangeD(py, size, -10, 10);
    initArrWithRandInRangeD(pz, size, -10, 10);

    initArrWithRandInRangeD(vx, size, -10, 10);
    initArrWithRandInRangeD(vy, size, -10, 10);
    initArrWithRandInRangeD(vz, size, -10, 10);

    initArrWithRandInRangeD(mass, size, 1, 1e9);
  }
};

#endif // NEON_SVE_BENCH_RANDOM_DATA_GENERATOR_H
