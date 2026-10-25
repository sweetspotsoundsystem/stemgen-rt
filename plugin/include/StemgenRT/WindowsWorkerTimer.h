#pragma once

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <chrono>
#include <cstdint>

namespace audio_plugin {

// Owned exclusively by a worker (or a standalone test host). No callback
// creates, arms, waits on, or destroys this timer. Unlike sleep_for on MSVC,
// this preserves sub-millisecond requests without changing the process's
// timer period. Scheduling can still delay a signaled thread.
class WindowsWorkerTimer {
 public:
  using Clock = std::chrono::steady_clock;

  WindowsWorkerTimer() noexcept
      : timer_(CreateWaitableTimerExW(nullptr,
                                      nullptr,
                                      CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                      TIMER_MODIFY_STATE | SYNCHRONIZE)) {}
  ~WindowsWorkerTimer() {
    if (timer_ != nullptr)
      CloseHandle(timer_);
  }
  WindowsWorkerTimer(const WindowsWorkerTimer&) = delete;
  WindowsWorkerTimer& operator=(const WindowsWorkerTimer&) = delete;

  bool isValid() const noexcept { return timer_ != nullptr; }

  bool waitFor(std::chrono::nanoseconds duration) noexcept {
    if (!isValid())
      return false;
    if (duration.count() <= 0)
      return true;
    // Round up to 100 ns ticks without overflowing the signed duration.
    const int64_t ns = duration.count();
    LARGE_INTEGER due{};
    due.QuadPart = -(ns / 100 + (ns % 100 != 0 ? 1 : 0));
    return SetWaitableTimerEx(timer_, &due, 0, nullptr, nullptr, nullptr, 0) !=
               FALSE &&
           WaitForSingleObject(timer_, INFINITE) == WAIT_OBJECT_0;
  }

  bool waitUntil(Clock::time_point deadline) noexcept {
    if (!isValid())
      return false;
    for (;;) {
      const auto now = Clock::now();
      if (now >= deadline)
        return true;
      if (!waitFor(std::chrono::duration_cast<std::chrono::nanoseconds>(
              deadline - now)))
        return false;
    }
  }

 private:
  HANDLE timer_{};
};

}  // namespace audio_plugin
#endif
