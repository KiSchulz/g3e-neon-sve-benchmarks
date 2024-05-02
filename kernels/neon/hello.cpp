#include "neon_common.h"

#include <unistd.h>

uint64_t neon_kernels::helloNeon() { return getpid(); }
