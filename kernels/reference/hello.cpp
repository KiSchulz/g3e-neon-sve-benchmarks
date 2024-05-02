#include "reference_common.h"

#include <unistd.h>

uint64_t reference_kernels::helloReference() {
  return getpid();
}