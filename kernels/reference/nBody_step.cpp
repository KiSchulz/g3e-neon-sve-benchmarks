#include "reference_common.h"

#include <cmath>

void reference_kernels::nBody_step(double *px, double *py, double *pz, double *vx, double *vy, double *vz,
                                   const double *m, double dt, std::size_t len) {
  for (std::size_t i = 0; i < len; i++) {
    double ax = 0, ay = 0, az = 0;
    for (std::size_t j = 0; j < len; j++) {
      // compute distance between the two bodies on each axis
      double dx = px[i] - px[j];
      double dy = py[i] - py[j];
      double dz = pz[i] - pz[j];

      double r2 = (dx * dx) + (dy * dy) + (dz * dz);
      // compute the Euclidean distance between the two bodies
      double r = std::sqrt(r2);

      // compute the acceleration body i experiences from body j
      double acc = physics::G * m[j] / r2;
      double ar = acc / r;

      // accumulate the acceleration that body i experiences
      ax += dx * ar;
      ay += dy * ar;
      az += dz * ar;
    }
    // apply the computed acceleration to the velocity of body i
    vx[i] += ax * dt;
    vy[i] += ay * dt;
    vz[i] += az * dt;

    // apply the updated velocity to the position of i
    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;
  }
}
