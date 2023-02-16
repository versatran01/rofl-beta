#include "sv/util/grid2d.h"

#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(Grid2dTest, TestEmptyGrid) {
  Grid2d<int> g;
  EXPECT_EQ(g.size(), 0);
  EXPECT_EQ(g.area(), 0);
  EXPECT_EQ(g.empty(), true);
}

TEST(Grid2dTest, TestGridCtor) {
  Grid2d<int> g(2, 3, 4);
  EXPECT_EQ(g.size(), 6);
  EXPECT_EQ(g.area(), 6);
  EXPECT_EQ(g.empty(), false);
  EXPECT_EQ(g.bytes(), 2 * 3 * 4);
  EXPECT_EQ(g.rows(), 2);
  EXPECT_EQ(g.cols(), 3);
}

}  // namespace
}  // namespace sv
