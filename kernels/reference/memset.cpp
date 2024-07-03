#include "reference_common.h"

#include <cstring>

void *reference_kernels::memset(void *dest, int ch, std::size_t count) { return std::memset(dest, ch, count); }