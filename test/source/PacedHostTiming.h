#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace audio_plugin_test {

struct PacedHostDelay {
  int callbackIndex{};
  double waitOverrunMicroseconds{};
  double preparationMicroseconds{};
  double bookkeepingMicroseconds{};
};

// Fixed storage, with no I/O or allocation while pacing. These are wall-time
// phases, not CPU measurements. Catch-up callbacks and all qualification
// failures remain counted by the original gates; no deadline is rebased.
class PacedHostTiming {
 public:
  using Clock = std::chrono::steady_clock;
  static constexpr size_t kCapacity = 64U;

  explicit PacedHostTiming(double periodMicroseconds)
      : periodMicroseconds_(periodMicroseconds) {}

  void observe(int callbackIndex,
               Clock::time_point scheduled,
               Clock::time_point waitStarted,
               Clock::time_point waitFinished,
               Clock::time_point processStarted,
               Clock::time_point previousProcessFinished) noexcept {
    const auto us = [](Clock::duration value) {
      return std::chrono::duration<double, std::micro>(value).count();
    };
    const PacedHostDelay event{
        callbackIndex, us(waitFinished - std::max(scheduled, waitStarted)),
        us(processStarted - waitFinished),
        us(waitStarted - previousProcessFinished)};
    maximumWaitOverrun_ =
        std::max(maximumWaitOverrun_, event.waitOverrunMicroseconds);
    maximumPreparation_ =
        std::max(maximumPreparation_, event.preparationMicroseconds);
    maximumBookkeeping_ =
        std::max(maximumBookkeeping_, event.bookkeepingMicroseconds);
    if (event.waitOverrunMicroseconds <= periodMicroseconds_ &&
        event.preparationMicroseconds <= periodMicroseconds_ &&
        event.bookkeepingMicroseconds <= periodMicroseconds_) {
      return;
    }
    ++delayCount_;
    if (size_ < events_.size()) {
      events_[size_++] = event;
    }
  }

  const std::array<PacedHostDelay, kCapacity>& events() const noexcept {
    return events_;
  }
  size_t size() const noexcept { return size_; }
  uint64_t delayCount() const noexcept { return delayCount_; }
  uint64_t omittedEvents() const noexcept { return delayCount_ - size_; }
  double maximumWaitOverrun() const noexcept { return maximumWaitOverrun_; }
  double maximumPreparation() const noexcept { return maximumPreparation_; }
  double maximumBookkeeping() const noexcept { return maximumBookkeeping_; }

 private:
  double periodMicroseconds_;
  std::array<PacedHostDelay, kCapacity> events_{};
  size_t size_{};
  uint64_t delayCount_{};
  double maximumWaitOverrun_{};
  double maximumPreparation_{};
  double maximumBookkeeping_{};
};

}  // namespace audio_plugin_test
