#include "StemgenRT/InferenceTiming.h"

#include <limits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <time.h>
#endif

namespace audio_plugin {

int64_t currentThreadCpuNanoseconds() noexcept {
#if defined(_WIN32)
  FILETIME creation{}, exit{}, kernel{}, user{};
  if (!GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user))
    return -1;
  const auto ticks = [](const FILETIME& value) {
    return (static_cast<uint64_t>(value.dwHighDateTime) << 32U) |
           static_cast<uint64_t>(value.dwLowDateTime);
  };
  constexpr auto limit =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 100U;
  const auto kernelTicks = ticks(kernel);
  const auto userTicks = ticks(user);
  if (kernelTicks > limit || userTicks > limit - kernelTicks)
    return -1;
  return static_cast<int64_t>((kernelTicks + userTicks) * 100U);
#elif defined(CLOCK_THREAD_CPUTIME_ID)
  timespec value{};
  constexpr int64_t billion = 1000000000;
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0 || value.tv_sec < 0 ||
      value.tv_nsec < 0 || value.tv_nsec >= billion ||
      value.tv_sec >
          (std::numeric_limits<int64_t>::max() - value.tv_nsec) / billion)
    return -1;
  return static_cast<int64_t>(value.tv_sec) * billion + value.tv_nsec;
#else
  return -1;
#endif
}

}  // namespace audio_plugin
