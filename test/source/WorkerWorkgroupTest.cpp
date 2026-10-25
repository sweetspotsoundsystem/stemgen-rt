#include <StemgenRT/WorkerWorkgroup.h>
#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <utility>

namespace {

thread_local bool audioThread = false;

struct Record {
  std::atomic<int> references{0}, joins{0}, leaves{0}, badReleases{0};
  std::atomic<int> invalidLeaves{0};
  std::thread::id joinedThread;
};

struct Token {
  Record* record{nullptr};
  ~Token() { reset(); }
  void reset() {
    if (record != nullptr) {
      if (audioThread || record->references.load() == 0 ||
          record->joinedThread != std::this_thread::get_id()) {
        ++record->invalidLeaves;
      }
      ++record->leaves;
      record = nullptr;
    }
  }
};

struct Context {
  Record* record{nullptr};
  Context() = default;
  explicit Context(Record& value) : record(&value) { ++record->references; }
  Context(const Context& other) : record(other.record) {
    if (record != nullptr) {
      ++record->references;
    }
  }
  Context(Context&& other) noexcept
      : record(std::exchange(other.record, nullptr)) {}
  Context& operator=(Context&& other) noexcept {
    release();
    record = std::exchange(other.record, nullptr);
    return *this;
  }
  ~Context() { release(); }
  void release() {
    if (record != nullptr) {
      if (audioThread) {
        ++record->badReleases;
      }
      --record->references;
      record = nullptr;
    }
  }
  bool operator!=(const Context& other) const { return record != other.record; }
  explicit operator bool() const { return record != nullptr; }
  void join(Token& token) const {
    if (token.record != record) {
      token.reset();
      token.record = record;
      record->joinedThread = std::this_thread::get_id();
      ++record->joins;
    }
  }
};

using Controller = audio_plugin::WorkerWorkgroupContext<Context, Token>;
using Status = audio_plugin::WorkerWorkgroupStatus;

bool publish(Controller& controller, const Context& context) {
  audioThread = true;
  const bool published = controller.publish(context);
  audioThread = false;
  return published;
}

TEST(WorkerWorkgroupTest, SupersededHandlesRetireOnWorkerAndOnlyLatestJoins) {
  Record a, b;
  const Context first(a), second(b);
  Controller controller;
  ASSERT_TRUE(publish(controller, first));
  ASSERT_TRUE(publish(controller, second));
  ASSERT_TRUE(publish(controller, second));
  std::thread worker([&] {
    controller.workerStarted(true);
    EXPECT_EQ(controller.getConsumedGeneration(), 3U);
    EXPECT_EQ(controller.getStatus(), Status::JoinRequested);
    controller.workerStopped();
  });
  worker.join();
  EXPECT_EQ(a.joins, 0);
  EXPECT_EQ(a.references, 1);
  EXPECT_EQ(b.joins, 1);
  EXPECT_EQ(b.leaves, 1);
  EXPECT_EQ(b.references, 2);  // Host plus retained restart context.
  EXPECT_EQ(a.badReleases + b.badReleases, 0);
  EXPECT_EQ(b.invalidLeaves, 0);
}

TEST(WorkerWorkgroupTest, RestartsRejoinWithoutAnotherHostNotification) {
  Record record;
  const Context context(record);
  Controller controller;
  ASSERT_TRUE(publish(controller, context));
  for (int restart = 0; restart < 3; ++restart) {
    std::thread worker([&] {
      controller.workerStarted(true);
      controller.poll();
      controller.workerStopped();
    });
    worker.join();
    EXPECT_EQ(record.joins, restart + 1);
    EXPECT_EQ(record.leaves, restart + 1);
  }
  EXPECT_EQ(record.invalidLeaves, 0);
  EXPECT_EQ(record.badReleases, 0);
}

TEST(WorkerWorkgroupTest, SaturationInvalidatesOldMembershipAndCanRecover) {
  Record old, next;
  const Context a(old), b(next);
  Controller controller;
  ASSERT_TRUE(publish(controller, a));
  // Model an in-flight run: the worker cannot consume the host's burst yet.
  std::thread worker([&] {
    controller.workerStarted(true);
    EXPECT_EQ(old.joins, 1);
    for (int update = 0; update < 8; ++update) {
      EXPECT_TRUE(publish(controller, b));
    }
    EXPECT_FALSE(publish(controller, b));
    EXPECT_EQ(controller.getOverflowCount(), 1U);
    controller.poll();
    EXPECT_EQ(controller.getStatus(), Status::Overflow);
    EXPECT_EQ(controller.getConsumedGeneration(), 10U);
    EXPECT_EQ(old.leaves, 1);
    EXPECT_EQ(old.references, 1);
    EXPECT_EQ(next.references, 1);
    EXPECT_EQ(next.joins, 0);
    EXPECT_TRUE(publish(controller, b));
    controller.poll();
    EXPECT_EQ(next.joins, 1);
    EXPECT_TRUE(publish(controller, Context{}));
    controller.poll();
    EXPECT_EQ(controller.getStatus(), Status::NoContext);
    EXPECT_EQ(next.leaves, 1);
    controller.workerStopped();
  });
  worker.join();
  EXPECT_EQ(old.badReleases + next.badReleases, 0);
  EXPECT_EQ(old.invalidLeaves + next.invalidLeaves, 0);
}

TEST(WorkerWorkgroupTest, OfflineWorkerRetainsContextWithoutJoining) {
  Record record;
  const Context context(record);
  Controller controller;
  ASSERT_TRUE(publish(controller, context));
  std::thread offline([&] {
    controller.workerStarted(false);
    EXPECT_EQ(controller.getStatus(), Status::Disabled);
    EXPECT_EQ(record.joins, 0);
    controller.workerStopped();
  });
  offline.join();
  std::thread playback([&] {
    controller.workerStarted(true);
    EXPECT_EQ(record.joins, 1);
    controller.workerStopped();
  });
  playback.join();
  EXPECT_EQ(record.invalidLeaves, 0);
}

TEST(WorkerWorkgroupTest, ConcurrentContextChurnPreservesOwnership) {
  Record a, b;
  const Context first(a), second(b);
  {
    Controller controller;
    std::atomic<bool> finished{false};
    std::thread worker([&] {
      controller.workerStarted(true);
      while (!finished.load(std::memory_order_acquire)) {
        controller.poll();
        std::this_thread::yield();
      }
      controller.poll();
      EXPECT_EQ(controller.getConsumedGeneration(), 20001U);
      controller.workerStopped();
    });
    for (int update = 0; update < 20000; ++update) {
      static_cast<void>(publish(controller, update % 2 == 0 ? first : second));
    }
    static_cast<void>(publish(controller, Context{}));
    finished.store(true, std::memory_order_release);
    worker.join();
  }
  EXPECT_EQ(a.references, 1);
  EXPECT_EQ(b.references, 1);
  EXPECT_EQ(a.joins, a.leaves);
  EXPECT_EQ(b.joins, b.leaves);
  EXPECT_EQ(a.badReleases + b.badReleases, 0);
  EXPECT_EQ(a.invalidLeaves + b.invalidLeaves, 0);
}

}  // namespace
