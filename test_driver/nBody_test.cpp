#include "common/random_data_generator.h"
#include "test_common.h"

#include <cstring>

using Func = void (*)(double *, double *, double *, double *, double *, double *, const double *, double, std::size_t);

template <bool fastMath> class NBodyTest : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
  constexpr static double dt = 0.01;

  std::size_t doubleEQ_n;
  std::size_t num_iterations;
  double relEQ_relDiv;
  double maxDeviation;

  static bool doubleEQ(double a, double b, std::size_t n) { return std::abs(a - b) <= (double)n * EPSILON_D; }
  static bool relEQ(double a, double b, double relDiv) { return std::abs(a - b) / std::min(a, b) <= relDiv; }

  NBodyTest() {
    doubleEQ_n = 1;
    num_iterations = 1024;
    relEQ_relDiv = 2e-5;
    maxDeviation = 1e-12;
  }

  static void singleNonMovingObject(Func f) {
    double pos[] = {0, 0, 0};
    double vel[] = {0, 0, 0};
    const double mass = 1;
    const std::size_t n = 1;

    f(&pos[0], &pos[1], &pos[2], &vel[0], &vel[1], &vel[2], &mass, dt, n);

    ASSERT_EQ(pos[0], 0.0);
    ASSERT_EQ(pos[1], 0.0);
    ASSERT_EQ(pos[2], 0.0);

    ASSERT_EQ(vel[0], 0.0);
    ASSERT_EQ(vel[1], 0.0);
    ASSERT_EQ(vel[2], 0.0);
  }

  static void singleConstVelObject(Func f) {
    double pos[] = {0, 0, 0};
    double vel[] = {1, 1, 1};
    const double mass = 1;
    const std::size_t n = 1;

    for (int i = 1; i < 10; i++) {
      f(&pos[0], &pos[1], &pos[2], &vel[0], &vel[1], &vel[2], &mass, dt, n);

      ASSERT_DOUBLE_EQ(pos[0], i * vel[0] * dt) << "failed after: " << i;
      ASSERT_DOUBLE_EQ(pos[1], i * vel[1] * dt) << "failed after: " << i;
      ASSERT_DOUBLE_EQ(pos[2], i * vel[2] * dt) << "failed after: " << i;

      ASSERT_EQ(vel[0], 1);
      ASSERT_EQ(vel[1], 1);
      ASSERT_EQ(vel[2], 1);
    }
  }

  void binarySystem(Func f) {
    double x_pos[] = {0, 1.0f};
    double y_pos[] = {0, 0};
    double z_pos[] = {0, 0};

    double x_vel[] = {0, 0};
    double y_vel[] = {0, 0};
    double z_vel[] = {0, 0};

    const double mass[] = {1e9f, 1};

    const std::size_t n = 2;

    f(x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, mass, dt, n);

    ASSERT_PRED2([&](double a, double b) { return fastMath ? relEQ(a, b, maxDeviation) : doubleEQ(a, b, doubleEQ_n); },
                 x_pos[0], physics::G * mass[1] * dt * dt);
    ASSERT_EQ(y_pos[0], 0);
    ASSERT_EQ(z_pos[0], 0);

    ASSERT_PRED2([&](double a, double b) { return fastMath ? relEQ(a, b, maxDeviation) : doubleEQ(a, b, doubleEQ_n); },
                 x_pos[1], 1.0f - physics::G * mass[0] * dt * dt);
    ASSERT_EQ(y_pos[1], 0);
    ASSERT_EQ(z_pos[1], 0);

    ASSERT_PRED2([&](double a, double b) { return fastMath ? relEQ(a, b, maxDeviation) : doubleEQ(a, b, doubleEQ_n); },
                 x_vel[0], physics::G * mass[1] * dt);
    ASSERT_EQ(y_vel[0], 0);
    ASSERT_EQ(z_vel[0], 0);

    ASSERT_PRED2([&](double a, double b) { return fastMath ? relEQ(a, b, maxDeviation) : doubleEQ(a, b, doubleEQ_n); },
                 x_vel[1], -physics::G * mass[0] * dt);
    ASSERT_EQ(y_vel[1], 0);
    ASSERT_EQ(z_vel[1], 0);
  }

  void systemEnergyConservation(Func f) {
    const std::size_t n = 17;

    double x_pos[n];
    double y_pos[n];
    double z_pos[n];

    double x_vel[n];
    double y_vel[n];
    double z_vel[n];

    double mass[n];

    initNBodySystem(x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, mass, n);

    auto computeEKin = [&]() {
      double E = 0;
      for (std::size_t i = 0; i < n; i++) {
        E += 0.5f * mass[i] * std::sqrt((x_vel[i] * x_vel[i]) + (y_vel[i] * y_vel[i]) + (z_vel[i] * z_vel[i]));
      }
      return E;
    };

    auto computeEPot = [&]() {
      double E = 0;
      for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = i + 1; j < n; j++) {
          double dx = x_pos[i] - x_pos[j];
          double dy = y_pos[i] - y_pos[j];
          double dz = z_pos[i] - z_pos[j];
          E += (physics::G * mass[i] * mass[j]) / (std::sqrt((dx * dx) + (dy * dy) + (dz * dz)));
        }
      }
      return E;
    };

    double initialE = computeEKin() + computeEPot();

    double last = initialE;
    for (int i = 0; i < 10; i++) {
      f(x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, mass, dt, n);

      double current = computeEKin() + computeEPot();
      ASSERT_PRED2([&](double a, double b) { return relEQ(a, b, relEQ_relDiv); }, initialE, current)
          << "failed after: " << i;
      ASSERT_PRED2([&](double a, double b) { return relEQ(a, b, relEQ_relDiv); }, last, current)
          << "failed after: " << i;
      last = current;
    }
  }

  void compareAgainstRef(Func f) {
    const std::size_t n = 127;
    auto *sysMem = new double[n * 7];
    double *px = sysMem;
    double *py = sysMem + n;
    double *pz = sysMem + 2 * n;
    double *vx = sysMem + 3 * n;
    double *vy = sysMem + 4 * n;
    double *vz = sysMem + 5 * n;
    double *m = sysMem + 6 * n;

    initNBodySystem(px, py, pz, vx, vy, vz, m, n);

    auto *sysMem2 = new double[n * 7];
    double *px2 = sysMem2;
    double *py2 = sysMem2 + n;
    double *pz2 = sysMem2 + 2 * n;
    double *vx2 = sysMem2 + 3 * n;
    double *vy2 = sysMem2 + 4 * n;
    double *vz2 = sysMem2 + 5 * n;
    double *m2 = sysMem2 + 6 * n;

    std::copy(sysMem, sysMem + n * 7, sysMem2);

    for (std::size_t i = 0; i < num_iterations; i++) {
      ref::nBody_step(px, py, pz, vx, vy, vz, m, dt, n);
      f(px2, py2, pz2, vx2, vy2, vz2, m2, dt, n);

      for (std::size_t j = 0; j < n; j++) {
        ASSERT_PRED2([&](double a, double b) { return relEQ(a, b, maxDeviation); }, px[j], px2[j]) << "failed: " << i;
        ASSERT_PRED2([&](double a, double b) { return relEQ(a, b, maxDeviation); }, py[j], py2[j]) << "failed: " << i;
        ASSERT_PRED2([&](double a, double b) { return relEQ(a, b, maxDeviation); }, pz[j], pz2[j]) << "failed: " << i;

        ASSERT_PRED2([&](double a, double b) { return relEQ(a, b, maxDeviation); }, vx[j], vx2[j]) << "failed: " << i;
        ASSERT_PRED2([&](double a, double b) { return relEQ(a, b, maxDeviation); }, vy[j], vy2[j]) << "failed: " << i;
        ASSERT_PRED2([&](double a, double b) { return relEQ(a, b, maxDeviation); }, vz[j], vz2[j]) << "failed: " << i;
      }
    }

    delete[] sysMem;
    delete[] sysMem2;
  }
};

using NBodyTest_normal = NBodyTest<false>;
using NBodyTest_fastMath = NBodyTest<true>;

TEST_P(NBodyTest_normal, singleNonMovingObject) { singleNonMovingObject(GetParam()); }
TEST_P(NBodyTest_normal, singleConstVelObject) { singleConstVelObject(GetParam()); }
TEST_P(NBodyTest_normal, binarySystem) { binarySystem(GetParam()); }
TEST_P(NBodyTest_normal, systemEnergyConservation) { systemEnergyConservation(GetParam()); }
TEST_P(NBodyTest_normal, compareAgainstRef) { compareAgainstRef(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Kernels, NBodyTest_normal,
                         testing::Values(&ref::nBody_step, &neon::nBody_step<false>, &sve::nBody_step<false>),
                         [](const auto &paramInfo) {
                           switch (paramInfo.index) {
                           case 0:
                             return "Ref";
                           case 1:
                             return "Neon";
                           case 2:
                             return "SVE";
                           default:
                             throw std::runtime_error("invalid argument was supplied");
                           }
                         });

TEST_P(NBodyTest_fastMath, singleNonMovingObject) { singleNonMovingObject(GetParam()); }
TEST_P(NBodyTest_fastMath, singleConstVelObject) { singleConstVelObject(GetParam()); }
TEST_P(NBodyTest_fastMath, binarySystem) { binarySystem(GetParam()); }
TEST_P(NBodyTest_fastMath, systemEnergyConservation) { systemEnergyConservation(GetParam()); }
TEST_P(NBodyTest_fastMath, compareAgainstRef) { compareAgainstRef(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Kernels, NBodyTest_fastMath,
                         testing::Values(&ref::nBody_step, &neon::nBody_step<true>, &sve::nBody_step<true>),
                         [](const auto &paramInfo) {
                           switch (paramInfo.index) {
                           case 0:
                             return "Ref";
                           case 1:
                             return "Neon";
                           case 2:
                             return "SVE";
                           default:
                             throw std::runtime_error("invalid argument was supplied");
                           }
                         });
