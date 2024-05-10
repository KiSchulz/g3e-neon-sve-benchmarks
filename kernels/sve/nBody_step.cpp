#include "sve_common.h"

template <bool fastMath>
void sve_kernels::nBody_step(double *px, double *py, double *pz, double *vx, double *vy, double *vz, const double *m,
                             double dt, std::size_t len) {
  const uint64_t vl = svcntd();
  const svfloat64_t vEpsilon = svdup_f64(EPSILON_D);
  const svfloat64_t vG = svdup_f64(physics::G);
  const svfloat64_t vDt = svdup_f64(dt);
  const svbool_t vTrue = svptrue_b64();

  for (std::size_t i = 0; i < len; i++) {
    svfloat64_t ax = svdup_f64(0), ay = svdup_f64(0), az = svdup_f64(0);
    const svfloat64_t px_i = svdup_f64(px[i]), py_i = svdup_f64(py[i]), pz_i = svdup_f64(pz[i]);

    for (std::size_t j = 0; j < len; j += vl) {
      const svbool_t pred = svwhilelt_b64_u64(j, len);
      const svfloat64_t px_j = svld1(pred, px + j);
      const svfloat64_t py_j = svld1(pred, py + j);
      const svfloat64_t pz_j = svld1(pred, pz + j);

      const svfloat64_t dx = svsub_f64_x(pred, px_j, px_i);
      const svfloat64_t dy = svsub_f64_x(pred, py_j, py_i);
      const svfloat64_t dz = svsub_f64_x(pred, pz_j, pz_i);

      svfloat64_t r2 = svmad_f64_x(pred, dx, dx, vEpsilon);
      r2 = svmad_f64_x(pred, dy, dy, r2);
      r2 = svmad_f64_x(pred, dz, dz, r2);

      const svfloat64_t m_j = svld1(pred, m + j);
      svfloat64_t acc = svmul_f64_x(pred, vG, m_j);

      svfloat64_t ar;
      if constexpr (fastMath) {
        const svfloat64_t r = svrsqrte_f64(r2);
        r2 = svrecpe_f64(r2);
        acc = svmul_f64_x(pred, acc, r2);
        ar = svmul_f64_z(pred, acc, r);
      } else {
        const svfloat64_t r = svsqrt_f64_x(pred, r2);
        acc = svdiv_f64_x(pred, acc, r2);
        ar = svdiv_f64_z(pred, acc, r);
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

  for (std::size_t i = 0; i < len; i += vl) {
    const svbool_t pred = svwhilelt_b64_u64(i, len);
    svfloat64_t px_i = svld1(pred, px + i);
    svfloat64_t py_i = svld1(pred, py + i);
    svfloat64_t pz_i = svld1(pred, pz + i);

    const svfloat64_t vx_i = svld1(pred, vx + i);
    const svfloat64_t vy_i = svld1(pred, vy + i);
    const svfloat64_t vz_i = svld1(pred, vz + i);

    px_i = svmad_f64_x(pred, vx_i, vDt, px_i);
    py_i = svmad_f64_x(pred, vy_i, vDt, py_i);
    pz_i = svmad_f64_x(pred, vz_i, vDt, pz_i);

    svst1(pred, px + i, px_i);
    svst1(pred, py + i, py_i);
    svst1(pred, pz + i, pz_i);
  }
}

template void sve_kernels::nBody_step<false>(double *px, double *py, double *pz, double *vx, double *vy, double *vz,
                                             const double *m, double dt, std::size_t len);
template void sve_kernels::nBody_step<true>(double *px, double *py, double *pz, double *vx, double *vy, double *vz,
                                            const double *m, double dt, std::size_t len);
