#pragma once

#include <chrono>
#include <cstdint>

namespace audio_plugin {

// Diagnostic only. CPU consumed by the calling thread, or -1 when unavailable.
// Never call from processBlock. This is distinct from process-wide std::clock.
int64_t currentThreadCpuNanoseconds() noexcept;

inline double cpuMicrosecondsBetween(int64_t finished,
                                     int64_t started) noexcept {
  return started < 0 || finished < started
             ? -1.0
             : static_cast<double>(finished - started) / 1000.0;
}

// Optional per-call evidence. Both CPU reads enclose the wall-clock reads;
// wall minus CPU is an approximation, not an exact descheduling measurement.
struct InferenceRunTiming {
  using Clock = std::chrono::steady_clock;
  Clock::time_point started;
  Clock::time_point finished;
  int64_t cpuStarted{-1};
  int64_t cpuFinished{-1};
  bool observed{};
};

}  // namespace audio_plugin
