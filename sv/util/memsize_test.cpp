#include "sv/util/memsize.h"

#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(Memsize, Ctor) {
  Memsize m;
  EXPECT_EQ(m.bytes(), 0);
}

TEST(Memsize, Factory) {
  EXPECT_EQ(Bytes(1).bytes(), 1);
  EXPECT_EQ(KiloBytes(2).bytes(), 2 * 1024);
  EXPECT_EQ(MegaBytes(3).bytes(), 3 * 1024 * 1024);
  EXPECT_EQ(GigaBytes(4).bytes(), 4L * 1024L * 1024L * 1024L);
}

TEST(Memsize, Format) {
  EXPECT_EQ(Bytes(1).Repr(), "1b");
  EXPECT_EQ(Bytes(2048).Repr(), "2kb");
  EXPECT_EQ(Bytes(2500).Repr(), "2.441kb");
  EXPECT_EQ(Bytes(100000).Repr(), "97.656kb");
}

}  // namespace
}  // namespace sv
