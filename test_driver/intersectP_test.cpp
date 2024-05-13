#include "common/random_data_generator.h"
#include "test_common.h"

using Func = void (*)(const Bounds3f *, const Vec3f *, const float *, const Vec3f *, const int *, int *);

class IntersectPTest : public testing::TestWithParam<std::tuple<Func, std::size_t>>, public RandomDataGenerator {
public:
  // Input class for all versions of intersectP
  struct Input {
    Bounds3f *box;
    Vec3f o;
    Vec3f dir;
    float tMax = INFINITY_F;

    Input() : box(new Bounds3f[getMaxIntersectPWidth()]), o(), dir() {}
    ~Input() { delete[] box; }

    static std::size_t getMaxIntersectPWidth() {
      return std::max(ref::intersectPWidth(), std::max(neon::intersectPWidth(), sve::intersectPWidth()));
    }

    void computeDerivedValues(int *dirIsNeg, Vec3f *invDir) {
      for (int i = 0; i < 3; i++) {
        dirIsNeg[i] = dir[i] < 0.0 ? 1 : 0;
      }
      *invDir = dir.normalize().invertElements();
    }

    void getResult(Func f, int *results) {
      int dirIsNeg[3];
      Vec3f invDir{};
      computeDerivedValues(dirIsNeg, &invDir);
      f(box, &o, &tMax, &invDir, dirIsNeg, results);
    }

    void testFirst(Func f, bool expected) {
      int *results = (int *)alloca(getMaxIntersectPWidth() * sizeof(int));
      getResult(f, results);
      ASSERT_EQ(expected, (bool)results[0]);
    }

    void testN(Func f, std::size_t width, int *expected) {
      int *results = (int *)alloca(getMaxIntersectPWidth() * sizeof(int));
      getResult(f, results);

      for (std::size_t i = 0; i < width; i++) {
        ASSERT_EQ((bool)expected[i], (bool)results[i]);
      }
    }
  };

  static void positiveDir(Func f) {
    Input i;
    i.box[0] = Bounds3f{Vec3f{-1, -1, -1}, Vec3f{1, 1, 1}};
    i.o = Vec3f{-2, 0, 0};

    std::vector<Vec3f> trueDirs = {{1, 0, 0}, {1, 0.5, 0}, {1, 0.5, 0.5}};
    std::vector<Vec3f> falseDirs = {{0, 1, 0}, {0.5, 1, 0}, {0.5, 1, 0.5}, {0, 0, 1}, {0.5, 0, 1}, {0.5, 0.5, 1}};

    for (auto el : trueDirs) {
      i.dir = el;
      i.testFirst(f, true);
    }
    for (auto el : falseDirs) {
      i.dir = el;
      i.testFirst(f, false);
    }
  }

  static void negativeDir(Func f) {
    Input i;
    i.box[0] = Bounds3f{Vec3f{-1, -1, -1}, Vec3f{1, 1, 1}};
    i.o = Vec3f{-2, 0, 0};

    std::vector<Vec3f> trueDirs = {{1, 0, 0}, {1, -0.5, 0}, {1, -0.5, -0.5}};
    std::vector<Vec3f> falseDirs = {{0, 1, 0},       {-0.5, 1, 0}, {-0.5, 1, -0.5}, {0, 0, 1}, {-0.5, 0, 1},
                                    {-0.5, -0.5, 1}, {-1, 0, 0},   {0, -1, 0},      {0, 0, -1}};

    for (auto el : trueDirs) {
      i.dir = el;
      i.testFirst(f, true);
    }
    for (auto el : falseDirs) {
      i.dir = el;
      i.testFirst(f, false);
    }
  }

  static void emptyAABB(Func f) {
    Input i;
    i.box[0] = Bounds3f{Vec3f{1, 1, 1}, Vec3f{-1, -1, -1}};
    i.o = Vec3f{-1, 0, 0};
    i.dir = Vec3f{1, 0, 0};

    i.testFirst(f, false);
  }

  static void tMax(Func f) {
    Input i;
    i.box[0] = Bounds3f{Vec3f{-1, -1, -1}, Vec3f{1, 1, 1}};
    i.o = Vec3f{-2, 0, 0};
    i.dir = Vec3f{1, 0, 0};

    i.testFirst(f, true);

    i.tMax = 1 * (1 + 3 * EPSILON_F);
    i.testFirst(f, true);

    i.tMax = 1 / (1 + 3 * EPSILON_F);
    i.testFirst(f, false);

    i.tMax = 0;
    i.testFirst(f, false);
  }

  void hitsAllRandomAABB(Func f) {
    for (std::size_t i = 0; i < 1024; i++) {
      Input input;
      input.box[0] = randomAABB(Bounds3f{Vec3f{-1, -1, -1}, Vec3f{1, 1, 1}});
      input.o = randomVec3f(Bounds3f{Vec3f{-1e9, -1e9, -1e9}, Vec3f{1e9, 1e9, 1e9}});
      input.dir = randomVec3f(input.box[0]) - input.o;

      input.testFirst(f, true);
    }
  }

  void compareFirstLaneAgainstRef(Func f) {
    const Bounds3f baseBox = {Vec3f{-1, -1, -1}, Vec3f{1, 1, 1}};
    for (std::size_t i = 0; i < 1024; i++) {
      Input input;
      input.box[0] = randomAABB(baseBox);
      input.o = randomVec3f(baseBox);
      input.dir = randomVec3f(baseBox);

      int result;
      input.getResult(&ref::intersectP, &result);

      input.testFirst(f, (bool)result);
    }
  }

  void compareAllLanesAgainstRef(Func f, std::size_t width) {
    const Bounds3f baseBox = {Vec3f{-1, -1, -1}, Vec3f{1, 1, 1}};
    for (std::size_t i = 0; i < 1024; i++) {
      int *resultsRef = (int *)alloca(width * sizeof(int));
      int dirIsNeg[3];
      Vec3f invDir{};

      Input input;
      input.o = randomVec3f(baseBox);
      input.dir = randomVec3f(baseBox);
      input.computeDerivedValues(dirIsNeg, &invDir);
      for (std::size_t j = 0; j < width; j++) {
        input.box[j] = randomAABB(baseBox);
        ref::intersectP(&input.box[j], &input.o, &input.tMax, &invDir, dirIsNeg, &resultsRef[j]);
      }

      input.testN(f, width, resultsRef);
    }
  }
};

TEST_P(IntersectPTest, positiveDir) { positiveDir(std::get<0>(GetParam())); }
TEST_P(IntersectPTest, negativeDir) { negativeDir(std::get<0>(GetParam())); }
TEST_P(IntersectPTest, emptyAABB) { emptyAABB(std::get<0>(GetParam())); }
TEST_P(IntersectPTest, tMax) { tMax(std::get<0>(GetParam())); }
TEST_P(IntersectPTest, hitsAllRandomAABB) { hitsAllRandomAABB(std::get<0>(GetParam())); }
TEST_P(IntersectPTest, compareFirstLaneAgainstRef) { compareFirstLaneAgainstRef(std::get<0>(GetParam())); }
TEST_P(IntersectPTest, compareAllLanesAgainstRef) {
  auto [f, width] = GetParam();
  compareAllLanesAgainstRef(f, width);
};

INSTANTIATE_TEST_SUITE_P(Kernels, IntersectPTest,
                         testing::Values(std::make_tuple(&ref::intersectP, ref::intersectPWidth()),
                                         std::make_tuple(&neon::intersectP, neon::intersectPWidth()),
                                         std::make_tuple(&sve::intersectP, sve::intersectPWidth())),
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
