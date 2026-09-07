#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace audio_plugin_test {

struct PacedFailureCounters {
  uint64_t dueBoundaryMisses{};
  uint64_t underrunSamples{};
  uint64_t lateDiscardEvents{};

  bool isAtLeast(const PacedFailureCounters& previous) const noexcept {
    return dueBoundaryMisses >= previous.dueBoundaryMisses &&
           underrunSamples >= previous.underrunSamples &&
           lateDiscardEvents >= previous.lateDiscardEvents;
  }

  PacedFailureCounters since(
      const PacedFailureCounters& previous) const noexcept {
    return {dueBoundaryMisses - previous.dueBoundaryMisses,
            underrunSamples - previous.underrunSamples,
            lateDiscardEvents - previous.lateDiscardEvents};
  }

  bool hasFailure() const noexcept {
    return dueBoundaryMisses != 0U || underrunSamples != 0U ||
           lateDiscardEvents != 0U;
  }
};

struct PacedFailureEvent {
  int callbackIndex{};
  bool duringWarmup{};
  PacedFailureCounters delta;
  double callbackSpacingMicroseconds{};
  double callbackStartLatenessMicroseconds{};
  double previousCallbackStartLatenessMicroseconds{};
};

// Test-only fixed storage. Observation never prints, allocates, resets a
// processor counter or changes any qualification gate. A later discard can
// occur on a different callback than the original missed due boundary.
class PacedQualificationTrace {
 public:
  static constexpr size_t kCapacity = 64U;

  explicit PacedQualificationTrace(int warmupCallbacks)
      : warmupCallbacks_(warmupCallbacks) {}

  void observe(int callbackIndex,
               PacedFailureCounters counters,
               double spacingMicroseconds,
               double latenessMicroseconds,
               double previousLatenessMicroseconds) noexcept {
    if (!counters.isAtLeast(totals_)) {
      countersMonotonic_ = false;
      return;
    }

    const auto delta = counters.since(totals_);
    const bool warmup = callbackIndex < warmupCallbacks_;
    totals_ = counters;
    if (warmup) {
      warmup_ = counters;
    }
    if (!delta.hasFailure()) {
      return;
    }
    if (size_ == events_.size()) {
      ++omittedEvents_;
      return;
    }
    events_[size_++] = {callbackIndex,
                        warmup,
                        delta,
                        spacingMicroseconds,
                        latenessMicroseconds,
                        previousLatenessMicroseconds};
  }

  PacedFailureCounters warmupCounters() const noexcept { return warmup_; }
  PacedFailureCounters measuredCounters() const noexcept {
    return totals_.since(warmup_);
  }
  const std::array<PacedFailureEvent, kCapacity>& events() const noexcept {
    return events_;
  }
  size_t size() const noexcept { return size_; }
  uint64_t omittedEvents() const noexcept { return omittedEvents_; }
  bool countersMonotonic() const noexcept { return countersMonotonic_; }

 private:
  int warmupCallbacks_{};
  std::array<PacedFailureEvent, kCapacity> events_{};
  size_t size_{};
  uint64_t omittedEvents_{};
  PacedFailureCounters totals_;
  PacedFailureCounters warmup_;
  bool countersMonotonic_{true};
};

}  // namespace audio_plugin_test
