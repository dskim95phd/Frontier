#include "frontier/scheduler/kv_block_accounting.h"
#include "tests/test_support.h"

namespace {

using frontier::RequestId;
using frontier::config::SchedulerConfig;
using frontier::scheduler::KvBlockAccounting;
using frontier::scheduler::KvBlockAccountingError;
using frontier::test::expect;
using frontier::test::expect_throws;

SchedulerConfig make_config() {
  SchedulerConfig config;
  config.block_size = 4;
  config.num_blocks = 4;
  config.watermark_blocks_fraction = 0.25;
  return config;
}

void test_new_and_running_reservations_follow_block_boundaries() {
  KvBlockAccounting blocks{make_config()};
  expect(
      blocks.can_reserve(RequestId{0}, 0, 4),
      "one prompt block must fit above watermark");
  blocks.reserve(RequestId{0}, 0, 4);
  expect(
      blocks.allocated_blocks(RequestId{0}) == 1,
      "initial allocation materializes scheduled tokens only");
  expect(
      blocks.can_reserve(RequestId{0}, 4, 1),
      "one token over boundary must reserve a new block");
  blocks.reserve(RequestId{0}, 4, 1);
  expect(
      blocks.allocated_blocks(RequestId{0}) == 2,
      "decode growth must cross block boundary exactly");
  expect(
      blocks.free(RequestId{0}) == 2 && blocks.empty(),
      "free must conserve all allocated blocks");
}

void test_watermark_and_overcommit_are_rejected() {
  KvBlockAccounting blocks{make_config()};
  blocks.reserve(RequestId{0}, 0, 8);
  blocks.reserve(RequestId{1}, 0, 4);
  expect(
      !blocks.can_reserve(RequestId{2}, 0, 4),
      "watermark must keep one block free");
  expect_throws<KvBlockAccountingError>(
      [&blocks] {
        blocks.reserve(RequestId{2}, 0, 4);
      },
      "reservation below watermark must fail");
}

}  // namespace

int main() {
  int failures = 0;
  failures += frontier::test::run(
      "KV reservations follow block boundaries",
      test_new_and_running_reservations_follow_block_boundaries);
  failures += frontier::test::run(
      "KV watermark and overcommit are rejected",
      test_watermark_and_overcommit_are_rejected);
  return failures == 0 ? 0 : 1;
}
