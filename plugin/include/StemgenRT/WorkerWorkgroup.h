#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include "RealtimeContextMailbox.h"

namespace audio_plugin {

enum class WorkerWorkgroupStatus : uint8_t {
  Stopped,
  Disabled,
  NoContext,
  JoinRequested,
  Overflow,
};

// The token never outlives its worker thread. The retained current context
// survives stop/start because a host need not resend an unchanged context.
// Parameterized so ownership and thread-affinity tests also run off macOS.
template <typename Workgroup, typename Token>
class WorkerWorkgroupContext {
 public:
  static_assert(std::atomic<WorkerWorkgroupStatus>::is_always_lock_free);

  bool publish(const Workgroup& workgroup) noexcept {
    return mailbox_.publish(workgroup);
  }

  // These three methods run only on the inference worker.
  void workerStarted(bool enabled) {
    enabled_ = enabled;
    if (!consume()) {
      joinCurrent();
    }
  }
  void poll() { static_cast<void>(consume()); }
  void workerStopped() {
    token_.reset();
    status_.store(WorkerWorkgroupStatus::Stopped, std::memory_order_release);
  }

  WorkerWorkgroupStatus getStatus() const noexcept {
    return status_.load(std::memory_order_acquire);
  }
  uint64_t getOverflowCount() const noexcept {
    return mailbox_.getOverflowCount();
  }
  uint64_t getConsumedGeneration() const noexcept {
    return consumedGeneration_.load(std::memory_order_acquire);
  }

 private:
  bool consume() {
    const bool changed = mailbox_.consumeLatest(
        [this](std::optional<Workgroup> next, bool overflow) {
          overflow_ = overflow;
          if (!next || current_ != *next) {
            // The token borrows its group. Leave before releasing ownership.
            token_.reset();
            current_ = next ? std::move(*next) : Workgroup{};
          }
          joinCurrent();
        });
    if (changed) {
      consumedGeneration_.store(mailbox_.consumedGeneration(),
                                std::memory_order_release);
    }
    return changed;
  }

  void joinCurrent() {
    WorkerWorkgroupStatus status = WorkerWorkgroupStatus::Disabled;
    if (enabled_) {
      if (current_) {
        current_.join(token_);
        // JUCE 8.0.6's token bool reports provider presence, including a
        // failed native join. Do not misreport this as verified membership.
        status = WorkerWorkgroupStatus::JoinRequested;
      } else {
        status = overflow_ ? WorkerWorkgroupStatus::Overflow
                           : WorkerWorkgroupStatus::NoContext;
      }
    }
    status_.store(status, std::memory_order_release);
  }

  RealtimeContextMailbox<Workgroup> mailbox_;
  Workgroup current_;
  Token token_;
  bool enabled_{false}, overflow_{false};
  std::atomic<WorkerWorkgroupStatus> status_{WorkerWorkgroupStatus::Stopped};
  std::atomic<uint64_t> consumedGeneration_{0};
};

using WorkerWorkgroup =
    WorkerWorkgroupContext<juce::AudioWorkgroup, juce::WorkgroupToken>;

}  // namespace audio_plugin
