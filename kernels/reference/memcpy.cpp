#include "reference_common.h"

#include <cstring>

void *reference_kernels::memcpy(void *dest, const void *src, std::size_t count) {
  return std::memcpy(dest, src, count);
}