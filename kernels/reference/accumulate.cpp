#include "reference_common.h"

#include <numeric>

uint64_t reference_kernels::accumulate(const uint64_t *arr, std::size_t len) {
  return std::accumulate(arr, arr + len, 0ul);
}