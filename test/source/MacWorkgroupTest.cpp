#include <StemgenRT/MacThreadPolicy.h>
#include <StemgenRT/WorkerWorkgroup.h>
#include <gtest/gtest.h>

#include <cerrno>
#include <thread>
#include "MacTestWorkgroup.h"

#if defined(__APPLE__)
namespace {

// On a fresh thread, EALREADY proves that the JUCE join established actual
// membership. Probe cleanup also lets the same test prove that leave worked.
int probeMembership(os_workgroup_t group) {
  os_workgroup_join_token_s token{};
  const int result = os_workgroup_join(group, &token);
  if (result == 0) {
    os_workgroup_leave(group, &token);
  }
  return result;
}

TEST(MacWorkgroupTest, NativeMembershipChangesAndLeavesBeforeWorkerExit) {
  audio_plugin_test::MacTestWorkgroup first, second;
  ASSERT_NE(first.get(), nullptr);
  ASSERT_NE(second.get(), nullptr);
  audio_plugin::WorkerWorkgroup controller;
  ASSERT_TRUE(controller.publish(first.context()));
  for (int restart = 0; restart < 2; ++restart) {
    std::thread worker([&] {
      audio_plugin::MacThreadPolicy policy;
      EXPECT_TRUE(policy.start(true, 2700000));
      controller.workerStarted(true);
      EXPECT_EQ(probeMembership(first.get()), EALREADY);
      EXPECT_TRUE(controller.publish(second.context()));
      controller.poll();
      EXPECT_EQ(probeMembership(second.get()), EALREADY);
      EXPECT_TRUE(controller.publish({}));
      controller.poll();
      EXPECT_EQ(probeMembership(first.get()), 0);
      EXPECT_TRUE(controller.publish(first.context()));
      controller.poll();
      EXPECT_EQ(probeMembership(first.get()), EALREADY);
      controller.workerStopped();
      EXPECT_EQ(probeMembership(first.get()), 0);
      EXPECT_TRUE(policy.finish());
    });
    worker.join();
  }
}

TEST(MacWorkgroupTest, CancelledContextDoesNotCreateMembershipOrBreakCleanup) {
  audio_plugin_test::MacTestWorkgroup cancelled, probe;
  ASSERT_NE(cancelled.get(), nullptr);
  ASSERT_NE(probe.get(), nullptr);
  os_workgroup_cancel(cancelled.get());
  audio_plugin::WorkerWorkgroup controller;
  ASSERT_TRUE(controller.publish(cancelled.context()));
  std::thread worker([&] {
    audio_plugin::MacThreadPolicy policy;
    EXPECT_TRUE(policy.start(true, 2700000));
    controller.workerStarted(true);
    EXPECT_EQ(probeMembership(probe.get()), 0);
    EXPECT_TRUE(controller.publish(probe.context()));
    controller.poll();
    EXPECT_EQ(probeMembership(probe.get()), EALREADY);
    controller.workerStopped();
    EXPECT_EQ(probeMembership(probe.get()), 0);
    EXPECT_TRUE(policy.finish());
  });
  worker.join();
}

TEST(MacWorkgroupTest, SeparateInstancesJoinTheSameHostContext) {
  audio_plugin_test::MacTestWorkgroup host;
  ASSERT_NE(host.get(), nullptr);
  std::array<audio_plugin::WorkerWorkgroup, 2> controllers;
  std::array<std::thread, 2> workers;
  std::atomic<int> attached{0};
  for (size_t i = 0; i < controllers.size(); ++i) {
    ASSERT_TRUE(controllers[i].publish(host.context()));
    workers[i] = std::thread([&, i] {
      audio_plugin::MacThreadPolicy policy;
      EXPECT_TRUE(policy.start(true, 2700000));
      controllers[i].workerStarted(true);
      EXPECT_EQ(probeMembership(host.get()), EALREADY);
      attached.fetch_add(1, std::memory_order_release);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(3);
      while (attached.load(std::memory_order_acquire) != 2 &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      EXPECT_EQ(attached.load(), 2);
      controllers[i].workerStopped();
      EXPECT_EQ(probeMembership(host.get()), 0);
      EXPECT_TRUE(policy.finish());
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
}

}  // namespace
#endif
