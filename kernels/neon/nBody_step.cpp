#include "neon_common.h"

#include <cmath>

template <bool fastMath>
void neon_kernels::nBody_step(double *px, double *py, double *pz, double *vx, double *vy, double *vz, const double *m,
                              double dt, std::size_t len) {
  constexpr std::size_t n = 2;
  constexpr std::size_t numEl = reg_width / sizeof(*px);
  const std::size_t vLen = len - (len % (numEl * n));
  const float64x2_t vEpsilon = vdupq_n_f64(EPSILON_D);
  const float64x2_t vG = vdupq_n_f64(physics::G);

  for (std::size_t i = 0; i < len; i++) {
    float64x2_t ax = vdupq_n_f64(0), ay = vdupq_n_f64(0), az = vdupq_n_f64(0);
    const float64x2_t px_i = vdupq_n_f64(px[i]), py_i = vdupq_n_f64(py[i]), pz_i = vdupq_n_f64(pz[i]);

    #pragma clang loop unroll_count(2)
    for (std::size_t j = 0; j < vLen; j += numEl * n) {
      const float64x2_t px_j0 = vld1q_f64(px + j + 0 * numEl);
      const float64x2_t px_j1 = vld1q_f64(px + j + 1 * numEl);
      const float64x2_t py_j0 = vld1q_f64(py + j + 0 * numEl);
      const float64x2_t py_j1 = vld1q_f64(py + j + 1 * numEl);
      const float64x2_t pz_j0 = vld1q_f64(pz + j + 0 * numEl);
      const float64x2_t pz_j1 = vld1q_f64(pz + j + 1 * numEl);

      const float64x2_t dx0 = vsubq_f64(px_j0, px_i);
      const float64x2_t dx1 = vsubq_f64(px_j1, px_i);
      const float64x2_t dy0 = vsubq_f64(py_j0, py_i);
      const float64x2_t dy1 = vsubq_f64(py_j1, py_i);
      const float64x2_t dz0 = vsubq_f64(pz_j0, pz_i);
      const float64x2_t dz1 = vsubq_f64(pz_j1, pz_i);

      float64x2_t rq0 = vfmaq_f64(vEpsilon, dx0, dx0);
      float64x2_t rq1 = vfmaq_f64(vEpsilon, dx1, dx1);
      rq0 = vfmaq_f64(rq0, dy0, dy0);
      rq1 = vfmaq_f64(rq1, dy1, dy1);
      rq0 = vfmaq_f64(rq0, dz0, dz0);
      rq1 = vfmaq_f64(rq1, dz1, dz1);

      const float64x2_t m_j0 = vld1q_f64(m + j + 0 * numEl);
      const float64x2_t m_j1 = vld1q_f64(m + j + 1 * numEl);
      float64x2_t acc0 = vmulq_f64(vG, m_j0);
      float64x2_t acc1 = vmulq_f64(vG, m_j1);

      float64x2_t ar0;
      float64x2_t ar1;
      if constexpr (fastMath) {
        float64x2_t ir0 = vrsqrteq_f64(rq0);
        float64x2_t ir1 = vrsqrteq_f64(rq1);
        ir0 = vmulq_f64(vrsqrtsq_f64(rq0, vmulq_f64(ir0, ir0)), ir0);
        ir1 = vmulq_f64(vrsqrtsq_f64(rq1, vmulq_f64(ir1, ir1)), ir1);
        ir0 = vmulq_f64(vrsqrtsq_f64(rq0, vmulq_f64(ir0, ir0)), ir0);
        ir1 = vmulq_f64(vrsqrtsq_f64(rq1, vmulq_f64(ir1, ir1)), ir1);
        ir0 = vmulq_f64(vrsqrtsq_f64(rq0, vmulq_f64(ir0, ir0)), ir0);
        ir1 = vmulq_f64(vrsqrtsq_f64(rq1, vmulq_f64(ir1, ir1)), ir1);

        float64x2_t irq0 = vmulq_f64(ir0, ir0);
        float64x2_t irq1 = vmulq_f64(ir1, ir1);

        acc0 = vmulq_f64(acc0, irq0);
        acc1 = vmulq_f64(acc1, irq1);
        ar0 = vmulq_f64(acc0, ir0);
        ar1 = vmulq_f64(acc1, ir1);
      } else {
        const float64x2_t r0 = vsqrtq_f64(rq0);
        const float64x2_t r1 = vsqrtq_f64(rq1);
        acc0 = vdivq_f64(acc0, rq0);
        acc1 = vdivq_f64(acc1, rq1);
        ar0 = vdivq_f64(acc0, r0);
        ar1 = vdivq_f64(acc1, r1);
      }

      ax = vfmaq_f64(ax, dx0, ar0);
      ax = vfmaq_f64(ax, dx1, ar1);
      ay = vfmaq_f64(ay, dy0, ar0);
      ay = vfmaq_f64(ay, dy1, ar1);
      az = vfmaq_f64(az, dz0, ar0);
      az = vfmaq_f64(az, dz1, ar1);
    }
    double sAx = vaddvq_f64(ax);
    double sAy = vaddvq_f64(ay);
    double sAz = vaddvq_f64(az);

    // scalar loop tail
    for(std::size_t j = vLen; j < len; j++) {
      const double dx = px[j] - px[i];
      const double dy = py[j] - py[i];
      const double dz = pz[j] - pz[i];

      const double r2 = (dx * dx) + (dy * dy) + (dz * dz) + EPSILON_D;
      const double r = std::sqrt(r2);
      const double acc = physics::G * m[j] / r2;
      const double ar = acc / r;

      sAx += dx * ar;
      sAy += dy * ar;
      sAz += dz * ar;
    }

    vx[i] += sAx * dt;
    vy[i] += sAy * dt;
    vz[i] += sAz * dt;
  }

  for (std::size_t i = 0; i < vLen; i += numEl * n) {
    float64x2_t px_i0 = vld1q_f64(px + i + 0 * numEl);
    float64x2_t px_i1 = vld1q_f64(px + i + 1 * numEl);
    float64x2_t py_i0 = vld1q_f64(py + i + 0 * numEl);
    float64x2_t py_i1 = vld1q_f64(py + i + 1 * numEl);
    float64x2_t pz_i0 = vld1q_f64(pz + i + 0 * numEl);
    float64x2_t pz_i1 = vld1q_f64(pz + i + 1 * numEl);

    const float64x2_t vx_i0 = vld1q_f64(vx + i + 0 * numEl);
    const float64x2_t vx_i1 = vld1q_f64(vx + i + 1 * numEl);
    const float64x2_t vy_i0 = vld1q_f64(vy + i + 0 * numEl);
    const float64x2_t vy_i1 = vld1q_f64(vy + i + 1 * numEl);
    const float64x2_t vz_i0 = vld1q_f64(vz + i + 0 * numEl);
    const float64x2_t vz_i1 = vld1q_f64(vz + i + 1 * numEl);

    px_i0 = vfmaq_n_f64(px_i0, vx_i0, dt);
    px_i1 = vfmaq_n_f64(px_i1, vx_i1, dt);
    py_i0 = vfmaq_n_f64(py_i0, vy_i0, dt);
    py_i1 = vfmaq_n_f64(py_i1, vy_i1, dt);
    pz_i0 = vfmaq_n_f64(pz_i0, vz_i0, dt);
    pz_i1 = vfmaq_n_f64(pz_i1, vz_i1, dt);

    vst1q_f64(px + i + 0 * numEl, px_i0);
    vst1q_f64(px + i + 1 * numEl, px_i1);
    vst1q_f64(py + i + 0 * numEl, py_i0);
    vst1q_f64(py + i + 1 * numEl, py_i1);
    vst1q_f64(pz + i + 0 * numEl, pz_i0);
    vst1q_f64(pz + i + 1 * numEl, pz_i1);
  }

  for(std::size_t i = vLen; i < len; i++) {
    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;
  }
}

template void neon_kernels::nBody_step<false>(double *px, double *py, double *pz, double *vx, double *vy, double *vz,
                                              const double *m, double dt, std::size_t len);
template void neon_kernels::nBody_step<true>(double *px, double *py, double *pz, double *vx, double *vy, double *vz,
                                             const double *m, double dt, std::size_t len);
