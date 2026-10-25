#include <StemgenRT/WindowsThreadPolicy.h>
#include <gtest/gtest.h>

#if defined(_WIN32)
namespace {
using audio_plugin::WindowsThreadPolicy;
using Status = WindowsThreadPolicy::Status;
struct FakePolicy {
  int priority = THREAD_PRIORITY_NORMAL;
  int joins = 0, leaves = 0;
  bool joinOk = true, taskPriorityOk = true, leaveOk = true, restoreOk = true;
  AVRT_PRIORITY requested = AVRT_PRIORITY_NORMAL;
};
thread_local FakePolicy* fake;
const WindowsThreadPolicy::Api api{
    [] { return fake->priority; },
    [](int value) {
      if (value == THREAD_PRIORITY_NORMAL && !fake->restoreOk)
        return false;
      fake->priority = value;
      return true;
    },
    [](DWORD* index) -> HANDLE {
      EXPECT_EQ(*index, 0U);
      ++fake->joins;
      if (!fake->joinOk)
        return nullptr;
      *index = 1;
      fake->priority = 17;
      return fake;
    },
    [](HANDLE, AVRT_PRIORITY priority) {
      fake->requested = priority;
      return fake->taskPriorityOk;
    },
    [](HANDLE task) {
      EXPECT_EQ(task, fake);
      ++fake->leaves;
      if (!fake->leaveOk)
        return false;
      fake->priority = THREAD_PRIORITY_HIGHEST;
      return true;
    }};

TEST(WindowsThreadPolicyTest, OfflineSkipsMmcssAndRestoresPriority) {
  FakePolicy state;
  fake = &state;
  {
    WindowsThreadPolicy policy(api);
    ASSERT_TRUE(policy.start(false));
    EXPECT_EQ(policy.status(), Status::Applied);
    EXPECT_EQ(state.priority, THREAD_PRIORITY_HIGHEST);
    EXPECT_FALSE(policy.start(true));
  }
  EXPECT_EQ(state.priority, THREAD_PRIORITY_NORMAL);
  EXPECT_EQ(state.joins, 0);
  EXPECT_EQ(state.leaves, 0);
}

TEST(WindowsThreadPolicyTest,
     RealtimeUnregistersOnceBeforeRestoringAndCanRestart) {
  FakePolicy state;
  fake = &state;
  {
    WindowsThreadPolicy policy(api);
    for (int i = 1; i <= 8; ++i) {
      ASSERT_TRUE(policy.start(true));
      EXPECT_EQ(policy.status(), Status::Applied);
      EXPECT_EQ(state.requested, AVRT_PRIORITY_HIGH);
      EXPECT_EQ(state.priority, 17);
      ASSERT_TRUE(policy.finish());
      ASSERT_TRUE(policy.finish());
      EXPECT_EQ(state.priority, THREAD_PRIORITY_NORMAL);
      EXPECT_EQ(state.joins, i);
      EXPECT_EQ(state.leaves, i);
    }
  }
  EXPECT_EQ(state.leaves, 8);
}

TEST(WindowsThreadPolicyTest, UnavailableServiceUsesReportedFallback) {
  FakePolicy state;
  state.joinOk = false;
  fake = &state;
  {
    WindowsThreadPolicy policy(api);
    ASSERT_TRUE(policy.start(true));
    EXPECT_EQ(policy.status(), Status::Unsupported);
    EXPECT_EQ(state.priority, THREAD_PRIORITY_HIGHEST);
    EXPECT_TRUE(policy.finish());
  }
  EXPECT_EQ(state.leaves, 0);
  EXPECT_EQ(state.priority, THREAD_PRIORITY_NORMAL);
}

TEST(WindowsThreadPolicyTest, RejectedRelativePriorityRevertsBeforeFallback) {
  FakePolicy state;
  state.taskPriorityOk = false;
  fake = &state;
  {
    WindowsThreadPolicy policy(api);
    ASSERT_TRUE(policy.start(true));
    EXPECT_EQ(policy.status(), Status::Unsupported);
    EXPECT_EQ(state.leaves, 1);
    EXPECT_EQ(state.priority, THREAD_PRIORITY_HIGHEST);
  }
  EXPECT_EQ(state.leaves, 1);
  EXPECT_EQ(state.priority, THREAD_PRIORITY_NORMAL);
}

TEST(WindowsThreadPolicyTest, CleanupFailuresRemainVisibleAndRetainOwnership) {
  FakePolicy state;
  fake = &state;
  WindowsThreadPolicy policy(api);
  ASSERT_TRUE(policy.start(true));
  state.leaveOk = false;
  EXPECT_FALSE(policy.finish());
  EXPECT_EQ(policy.status(), Status::Failed);
  EXPECT_EQ(state.priority, 17);
  EXPECT_FALSE(policy.start(false));
  state.leaveOk = true;
  state.restoreOk = false;
  EXPECT_FALSE(policy.finish());
  EXPECT_EQ(state.priority, THREAD_PRIORITY_HIGHEST);
  state.restoreOk = true;
  EXPECT_FALSE(
      policy.finish());  // Cleanup succeeded; the prior failure persists.
  EXPECT_EQ(state.priority, THREAD_PRIORITY_NORMAL);
  EXPECT_EQ(state.leaves, 2);
}

TEST(WindowsThreadPolicyTest,
     NativeRegistrationOrFallbackRestoresCallingThread) {
  const int original = GetThreadPriority(GetCurrentThread());
  WindowsThreadPolicy policy;
  ASSERT_TRUE(policy.start(true));
  EXPECT_TRUE(policy.status() == Status::Applied ||
              policy.status() == Status::Unsupported);
  EXPECT_TRUE(policy.finish());
  EXPECT_EQ(GetThreadPriority(GetCurrentThread()), original);
}
}  // namespace
#endif
