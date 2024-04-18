#include "sve_common.h"

#include <unistd.h>

uint64_t sve_kernels::helloSVE() { return getpid(); }
