#include "test_common.h"

#include "common/random_data_generator.h"

using Func = void (*)(double *, double *, double *, double *, double *, double *, const double *, double, std::size_t);

class NBodyTest : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
  constexpr static double dt = 0.01;

  static bool doubleEQ(double a, double b, std::size_t n = 1) { return std::abs(a - b) <= (double)n * EPSILON_D; }
  static bool relEQ(double a, double b, double relDiv = 3e-4) { return std::abs(a - b) / std::min(a, b) <= relDiv; }

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

  static void binarySystem(Func f) {
    double x_pos[] = {0, 1.0f};
    double y_pos[] = {0, 0};
    double z_pos[] = {0, 0};

    double x_vel[] = {0, 0};
    double y_vel[] = {0, 0};
    double z_vel[] = {0, 0};

    const double mass[] = {1e9f, 1};

    const std::size_t n = 2;

    f(x_pos, y_pos, z_pos, x_vel, y_vel, z_vel, mass, dt, n);

    ASSERT_DOUBLE_EQ(x_pos[0], physics::G * mass[1] * dt * dt);
    ASSERT_EQ(y_pos[0], 0);
    ASSERT_EQ(z_pos[0], 0);

    ASSERT_DOUBLE_EQ(x_pos[1], 1.0f - physics::G * mass[0] * dt * dt);
    ASSERT_EQ(y_pos[1], 0);
    ASSERT_EQ(z_pos[1], 0);

    ASSERT_PRED2([&](double a, double b) { return doubleEQ(a, b); }, x_vel[0], physics::G * mass[1] * dt);
    ASSERT_EQ(y_vel[0], 0);
    ASSERT_EQ(z_vel[0], 0);

    ASSERT_PRED2([&](double a, double b) { return doubleEQ(a, b); }, x_vel[1], -physics::G * mass[0] * dt);
    ASSERT_EQ(y_vel[1], 0);
    ASSERT_EQ(z_vel[1], 0);
  }

  void systemEnergyConservation(Func f) {
    const std::size_t n = 17;

    double x_pos[n];
    double y_pos[n];
    double z_pos[n];
    initArrWithRandInRangeD(x_pos, n, -10, 10);
    initArrWithRandInRangeD(y_pos, n, -10, 10);
    initArrWithRandInRangeD(z_pos, n, -10, 10);

    double x_vel[n];
    double y_vel[n];
    double z_vel[n];
    initArrWithRandInRangeD(x_vel, n, -10, 10);
    initArrWithRandInRangeD(y_vel, n, -10, 10);
    initArrWithRandInRangeD(z_vel, n, -10, 10);

    double mass[n];
    initArrWithRandInRangeD(mass, n, 1, 1e9);

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
      ASSERT_PRED2([&](double a, double b) { return relEQ(a, b); }, initialE, current) << "failed after: " << i;
      ASSERT_PRED2([&](double a, double b) { return relEQ(a, b); }, last, current) << "failed after: " << i;
      last = current;
    }
  }
};

TEST_P(NBodyTest, singleNonMovingObject) { singleNonMovingObject(GetParam()); }
TEST_P(NBodyTest, singleConstVelObject) { singleConstVelObject(GetParam()); }
TEST_P(NBodyTest, binarySystem) { binarySystem(GetParam()); }
TEST_P(NBodyTest, systemEnergyConservation) { systemEnergyConservation(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Kernels, NBodyTest, testing::Values(&ref::nBody_step, &neon::nBody_step, &sve::nBody_step),
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
