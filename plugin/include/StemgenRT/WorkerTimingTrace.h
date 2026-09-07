#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <span>
#include <vector>

namespace audio_plugin {

struct WorkerTimingSample {
  using Clock = std::chrono::steady_clock;
  uint32_t epoch{};
  uint64_t inputSequence{};
  Clock::time_point acquired;
  Clock::time_point runStarted;
  Clock::time_point runFinished;
  Clock::time_point publishStarted;
  Clock::time_point publishFinished;
  // Total process CPU time on supported platforms, including other ORT and
  // synthetic-host threads. It is not the dedicated worker's CPU time alone.
  std::clock_t processCpuStarted{static_cast<std::clock_t>(-1)};
  std::clock_t processCpuFinished{static_cast<std::clock_t>(-1)};
  bool inferenceOk{};

  double processCpuMicroseconds() const noexcept {
    if (processCpuStarted == static_cast<std::clock_t>(-1) ||
        processCpuFinished == static_cast<std::clock_t>(-1) ||
        processCpuFinished < processCpuStarted) {
      return -1.0;
    }
    return 1.0e6 * static_cast<double>(processCpuFinished - processCpuStarted) /
           static_cast<double>(CLOCKS_PER_SEC);
  }
};

// Optional diagnostic storage, allocated before the worker starts. Only the
// worker writes; the owner may inspect it after stopThread() has joined. The
// storage must outlive the queue, or be detached while the worker is stopped.
// No allocation, printing or audio-thread instrumentation occurs per request.
class WorkerTimingTrace {
 public:
  explicit WorkerTimingTrace(size_t capacity) : samples_(capacity) {}
  WorkerTimingTrace(const WorkerTimingTrace&) = delete;
  WorkerTimingTrace& operator=(const WorkerTimingTrace&) = delete;

  void record(const WorkerTimingSample& sample) noexcept {
    if (size_ < samples_.size()) {
      samples_[size_++] = sample;
    } else {
      ++omitted_;
    }
  }

  std::span<const WorkerTimingSample> samples() const noexcept {
    return {samples_.data(), size_};
  }
  uint64_t omitted() const noexcept { return omitted_; }

 private:
  std::vector<WorkerTimingSample> samples_;
  size_t size_{};
  uint64_t omitted_{};
};

}  // namespace audio_plugin
