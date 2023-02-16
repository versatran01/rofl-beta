#include "sv/rofl/pwin.h"

#include <gtest/gtest.h>

namespace sv::rofl {
namespace {

TEST(Window, Ctor) {
  PanoWindow pwin(2, {1024, 256});
  EXPECT_EQ(pwin.size(), 0);
  EXPECT_EQ(pwin.empty(), true);
  EXPECT_EQ(pwin.full(), false);
  EXPECT_EQ(pwin.capacity(), 2);
}

TEST(Window, AddAndRemove) {
  PanoWindow pwin(2, {1024, 256});

  pwin.AddPano(0, 1, {});
  pwin.AddPano(1, 2, {});
  EXPECT_EQ(pwin.size(), 2);
  EXPECT_EQ(pwin.full(), true);
  EXPECT_EQ(pwin.At(0).id(), 0);
  EXPECT_EQ(pwin.At(1).id(), 1);
  EXPECT_EQ(pwin.first().id(), 0);
  EXPECT_EQ(pwin.last().id(), 1);
  EXPECT_EQ(pwin.removed().id(), -1);

  pwin.RemoveFront();
  EXPECT_EQ(pwin.size(), 1);
  EXPECT_EQ(pwin.full(), false);

  pwin.AddPano(2, 3, {});
  EXPECT_EQ(pwin.size(), 2);
  EXPECT_EQ(pwin.full(), true);
  EXPECT_EQ(pwin.At(0).id(), 1);
  EXPECT_EQ(pwin.At(1).id(), 2);
}

}  // namespace
}  // namespace sv::rofl
