#include "reference_common.h"

#include <cstring>

int reference_kernels::memcmp(const void *lhs, const void *rhs, std::size_t count) {
  return std::memcmp(lhs, rhs, count);
}