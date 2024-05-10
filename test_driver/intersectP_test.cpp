#include "common/random_data_generator.h"
#include "test_common.h"

using Func = void (*)(const Bounds3f *, const Vec3f *, const float *, const Vec3f *, const int *, bool *);

class IntersectPTest : public testing::TestWithParam<Func>, public RandomDataGenerator {
public:
  struct Input {
    Bounds3f box;
    Vec3f o;
    Vec3f dir;
    float tMax = INFINITY_F;

    void test(Func f, bool expected) {
      int dirIsNeg[3];
      for (int i = 0; i < 3; i++) {
        dirIsNeg[i] = dir[i] < 0.0 ? 1 : 0;
      }
      Vec3f invDir = dir.normalize().invertElements();
      bool result = !expected;
      f(&box, &o, &tMax, &invDir, dirIsNeg, &result);
      ASSERT_EQ(expected, result);
    }
  };

  static void positiveDir(Func f) {
    Input i;
    i.box = Bounds3f{Vec3f{-1, -1, -1}, Vec3f{1, 1, 1}};
    i.o = Vec3f{-2, 0, 0};

    std::vector<Vec3f> trueDirs = {{1, 0, 0}, {1, 0.5, 0}, {1, 0.5, 0.5}};
    std::vector<Vec3f> falseDirs = {{0, 1, 0}, {0.5, 1, 0}, {0.5, 1, 0.5}, {0, 0, 1}, {0.5, 0, 1}, {0.5, 0.5, 1}};

    for (auto el : trueDirs) {
      i.dir = el;
      i.test(f, true);
    }
    for (auto el : falseDirs) {
      i.dir = el;
      i.test(f, false);
    }
  }

  static void negativeDir(Func f) {
    Input i;
    i.box = Bounds3f{Vec3f{-1, -1, -1}, Vec3f{1, 1, 1}};
    i.o = Vec3f{-2, 0, 0};

    std::vector<Vec3f> trueDirs = {{1, 0, 0}, {1, -0.5, 0}, {1, -0.5, -0.5}};
    std::vector<Vec3f> falseDirs = {{0, 1, 0},       {-0.5, 1, 0}, {-0.5, 1, -0.5}, {0, 0, 1}, {-0.5, 0, 1},
                                    {-0.5, -0.5, 1}, {-1, 0, 0},   {0, -1, 0},      {0, 0, -1}};

    for (auto el : trueDirs) {
      i.dir = el;
      i.test(f, true);
    }
    for (auto el : falseDirs) {
      i.dir = el;
      i.test(f, false);
    }
  }

  static void emptyAABB(Func f) {
    Input i;
    i.box = Bounds3f{Vec3f{1, 1, 1}, Vec3f{-1, -1, -1}};
    i.o = Vec3f{-1, 0, 0};
    i.dir = Vec3f{1, 0, 0};

    i.test(f, false);
  }

  static void tMax(Func f) {
    Input i;
    i.box = Bounds3f{Vec3f{-1, -1, -1}, Vec3f{1, 1, 1}};
    i.o = Vec3f{-2, 0, 0};
    i.dir = Vec3f{1, 0, 0};

    i.test(f, true);

    i.tMax = 1 * (1 + 3 * EPSILON_F);
    i.test(f, true);

    i.tMax = 1 / (1 + 3 * EPSILON_F);
    i.test(f, false);

    i.tMax = 0;
    i.test(f, false);
  }

  static void hitsAllRandomAABB(Func f) {}

  static void compareAgainstRef(Func f) {}
};

TEST_P(IntersectPTest, positiveDir) { positiveDir(GetParam()); }
TEST_P(IntersectPTest, negativeDir) { negativeDir(GetParam()); }
TEST_P(IntersectPTest, emptyAABB) { emptyAABB(GetParam()); }
TEST_P(IntersectPTest, tMax) { tMax(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Kernels, IntersectPTest,
                         testing::Values(&ref::intersectP, &neon::intersectP, &sve::intersectP),
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
