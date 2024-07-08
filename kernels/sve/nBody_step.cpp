#include "sve_common.h"

template <bool fastMath>
void sve_kernels::nBody_step(double *px, double *py, double *pz, double *vx, double *vy, double *vz, const double *m,
                             double dt, std::size_t len) {
  constexpr uint64_t n = 2;
  const uint64_t vl = svcntd();
  const uint64_t vLen = len - (len % (vl * n));
  const svfloat64_t vEpsilon = svdup_f64(EPSILON_D);
  const svfloat64_t vG = svdup_f64(physics::G);
  const svfloat64_t vDt = svdup_f64(dt);
  const svbool_t vTrue = svptrue_b64();

  for (std::size_t i = 0; i < len; i++) {
    svfloat64_t ax = svdup_f64(0), ay = svdup_f64(0), az = svdup_f64(0);
    const svfloat64_t px_i = svdup_f64(px[i]), py_i = svdup_f64(py[i]), pz_i = svdup_f64(pz[i]);

    svbool_t p0 = svptrue_b64();
#pragma clang loop unroll_count(2)
    for (std::size_t j = 0; j < vLen; j += vl * n) {
      const svfloat64_t px_j0 = svld1(p0, px + j + 0 * vl);
      const svfloat64_t px_j1 = svld1(p0, px + j + 1 * vl);
      const svfloat64_t py_j0 = svld1(p0, py + j + 0 * vl);
      const svfloat64_t py_j1 = svld1(p0, py + j + 1 * vl);
      const svfloat64_t pz_j0 = svld1(p0, pz + j + 0 * vl);
      const svfloat64_t pz_j1 = svld1(p0, pz + j + 1 * vl);

      const svfloat64_t dx0 = svsub_f64_x(p0, px_j0, px_i);
      const svfloat64_t dx1 = svsub_f64_x(p0, px_j1, px_i);
      const svfloat64_t dy0 = svsub_f64_x(p0, py_j0, py_i);
      const svfloat64_t dy1 = svsub_f64_x(p0, py_j1, py_i);
      const svfloat64_t dz0 = svsub_f64_x(p0, pz_j0, pz_i);
      const svfloat64_t dz1 = svsub_f64_x(p0, pz_j1, pz_i);

      svfloat64_t rq0 = svmad_f64_x(p0, dx0, dx0, vEpsilon);
      svfloat64_t rq1 = svmad_f64_x(p0, dx1, dx1, vEpsilon);
      rq0 = svmad_f64_x(p0, dy0, dy0, rq0);
      rq1 = svmad_f64_x(p0, dy1, dy1, rq1);
      rq0 = svmad_f64_x(p0, dz0, dz0, rq0);
      rq1 = svmad_f64_x(p0, dz1, dz1, rq1);

      const svfloat64_t m_j0 = svld1(p0, m + j + 0 * vl);
      const svfloat64_t m_j1 = svld1(p0, m + j + 1 * vl);
      svfloat64_t acc0 = svmul_f64_x(p0, vG, m_j0);
      svfloat64_t acc1 = svmul_f64_x(p0, vG, m_j1);

      svfloat64_t ar0;
      svfloat64_t ar1;
      if constexpr (fastMath) {
        svfloat64_t ir0 = svrsqrte_f64(rq0);
        svfloat64_t ir1 = svrsqrte_f64(rq1);
        ir0 = svmul_f64_x(p0, svrsqrts_f64(rq0, svmul_f64_x(p0, ir0, ir0)), ir0);
        ir1 = svmul_f64_x(p0, svrsqrts_f64(rq1, svmul_f64_x(p0, ir1, ir1)), ir1);
        ir0 = svmul_f64_x(p0, svrsqrts_f64(rq0, svmul_f64_x(p0, ir0, ir0)), ir0);
        ir1 = svmul_f64_x(p0, svrsqrts_f64(rq1, svmul_f64_x(p0, ir1, ir1)), ir1);
        ir0 = svmul_f64_x(p0, svrsqrts_f64(rq0, svmul_f64_x(p0, ir0, ir0)), ir0);
        ir1 = svmul_f64_x(p0, svrsqrts_f64(rq1, svmul_f64_x(p0, ir1, ir1)), ir1);

        svfloat64_t irq0 = svmul_f64_x(p0, ir0, ir0);
        svfloat64_t irq1 = svmul_f64_x(p0, ir1, ir1);

        acc0 = svmul_f64_x(p0, acc0, irq0);
        acc1 = svmul_f64_x(p0, acc1, irq1);
        ar0 = svmul_f64_z(p0, acc0, ir0);
        ar1 = svmul_f64_z(p0, acc1, ir1);
      } else {
        const svfloat64_t r0 = svsqrt_f64_x(p0, rq0);
        const svfloat64_t r1 = svsqrt_f64_x(p0, rq1);
        acc0 = svdiv_f64_x(p0, acc0, rq0);
        acc1 = svdiv_f64_x(p0, acc1, rq1);
        ar0 = svdiv_f64_z(p0, acc0, r0);
        ar1 = svdiv_f64_z(p0, acc1, r1);
      }

      ax = svmad_f64_x(p0, dx0, ar0, ax);
      ax = svmad_f64_x(p0, dx1, ar1, ax);
      ay = svmad_f64_x(p0, dy0, ar0, ay);
      ay = svmad_f64_x(p0, dy1, ar1, ay);
      az = svmad_f64_x(p0, dz0, ar0, az);
      az = svmad_f64_x(p0, dz1, ar1, az);
    }

    for (std::size_t j = vLen; j < len; j += vl) {
      p0 = svwhilelt_b64_u64(j, len);
      const svfloat64_t px_j = svld1(p0, px + j);
      const svfloat64_t py_j = svld1(p0, py + j);
      const svfloat64_t pz_j = svld1(p0, pz + j);

      const svfloat64_t dx = svsub_f64_x(p0, px_j, px_i);
      const svfloat64_t dy = svsub_f64_x(p0, py_j, py_i);
      const svfloat64_t dz = svsub_f64_x(p0, pz_j, pz_i);

      svfloat64_t r2 = svmad_f64_x(p0, dx, dx, vEpsilon);
      r2 = svmad_f64_x(p0, dy, dy, r2);
      r2 = svmad_f64_x(p0, dz, dz, r2);

      const svfloat64_t m_j = svld1(p0, m + j);
      svfloat64_t acc = svmul_f64_x(p0, vG, m_j);

      svfloat64_t ar;
      if constexpr (fastMath) {
        svfloat64_t ir = svrsqrte_f64(r2);
        ir = svmul_f64_x(p0, svrsqrts_f64(r2, svmul_f64_x(p0, ir, ir)), ir);
        ir = svmul_f64_x(p0, svrsqrts_f64(r2, svmul_f64_x(p0, ir, ir)), ir);
        ir = svmul_f64_x(p0, svrsqrts_f64(r2, svmul_f64_x(p0, ir, ir)), ir);

        svfloat64_t ir2 = svmul_f64_x(p0, ir, ir);

        acc = svmul_f64_x(p0, acc, ir2);
        ar = svmul_f64_z(p0, acc, ir);
      } else {
        const svfloat64_t r = svsqrt_f64_x(p0, r2);
        acc = svdiv_f64_x(p0, acc, r2);
        ar = svdiv_f64_z(p0, acc, r);
      }

      ax = svmad_f64_x(vTrue, dx, ar, ax);
      ay = svmad_f64_x(vTrue, dy, ar, ay);
      az = svmad_f64_x(vTrue, dz, ar, az);
    }
    double sAx = svadda(vTrue, 0, ax);
    double sAy = svadda(vTrue, 0, ay);
    double sAz = svadda(vTrue, 0, az);

    vx[i] += sAx * dt;
    vy[i] += sAy * dt;
    vz[i] += sAz * dt;
  }

  svbool_t p0 = svptrue_b64();
  for (std::size_t i = 0; i < vLen; i += vl * n) {
    svfloat64_t px_i0 = svld1(p0, px + i + 0 * vl);
    svfloat64_t px_i1 = svld1(p0, px + i + 1 * vl);
    svfloat64_t py_i0 = svld1(p0, py + i + 0 * vl);
    svfloat64_t py_i1 = svld1(p0, py + i + 1 * vl);
    svfloat64_t pz_i0 = svld1(p0, pz + i + 0 * vl);
    svfloat64_t pz_i1 = svld1(p0, pz + i + 1 * vl);

    const svfloat64_t vx_i0 = svld1(p0, vx + i + 0 * vl);
    const svfloat64_t vx_i1 = svld1(p0, vx + i + 1 * vl);
    const svfloat64_t vy_i0 = svld1(p0, vy + i + 0 * vl);
    const svfloat64_t vy_i1 = svld1(p0, vy + i + 1 * vl);
    const svfloat64_t vz_i0 = svld1(p0, vz + i + 0 * vl);
    const svfloat64_t vz_i1 = svld1(p0, vz + i + 1 * vl);

    px_i0 = svmad_f64_x(p0, vx_i0, vDt, px_i0);
    px_i1 = svmad_f64_x(p0, vx_i1, vDt, px_i1);
    py_i0 = svmad_f64_x(p0, vy_i0, vDt, py_i0);
    py_i1 = svmad_f64_x(p0, vy_i1, vDt, py_i1);
    pz_i0 = svmad_f64_x(p0, vz_i0, vDt, pz_i0);
    pz_i1 = svmad_f64_x(p0, vz_i1, vDt, pz_i1);

    svst1(p0, px + i + 0 * vl, px_i0);
    svst1(p0, px + i + 1 * vl, px_i1);
    svst1(p0, py + i + 0 * vl, py_i0);
    svst1(p0, py + i + 1 * vl, py_i1);
    svst1(p0, pz + i + 0 * vl, pz_i0);
    svst1(p0, pz + i + 1 * vl, pz_i1);
  }

  for (std::size_t i = vLen; i < len; i += vl) {
    p0 = svwhilelt_b64_u64(i, len);
    svfloat64_t px_i = svld1(p0, px + i);
    svfloat64_t py_i = svld1(p0, py + i);
    svfloat64_t pz_i = svld1(p0, pz + i);

    const svfloat64_t vx_i = svld1(p0, vx + i);
    const svfloat64_t vy_i = svld1(p0, vy + i);
    const svfloat64_t vz_i = svld1(p0, vz + i);

    px_i = svmad_f64_x(p0, vx_i, vDt, px_i);
    py_i = svmad_f64_x(p0, vy_i, vDt, py_i);
    pz_i = svmad_f64_x(p0, vz_i, vDt, pz_i);

    svst1(p0, px + i, px_i);
    svst1(p0, py + i, py_i);
    svst1(p0, pz + i, pz_i);
  }
}

template void sve_kernels::nBody_step<false>(double *px, double *py, double *pz, double *vx, double *vy, double *vz,
                                             const double *m, double dt, std::size_t len);
template void sve_kernels::nBody_step<true>(double *px, double *py, double *pz, double *vx, double *vy, double *vz,
                                            const double *m, double dt, std::size_t len);
