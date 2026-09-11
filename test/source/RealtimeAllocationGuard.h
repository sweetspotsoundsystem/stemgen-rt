#pragma once

#include <cstddef>

namespace audio_plugin_test {

struct HeapTraffic {
  size_t allocations{0};
  size_t deallocations{0};
};

// Counts this thread's C++ new/delete traffic only, including aligned forms.
// It does not intercept C malloc/free or allocations inside shared libraries.
class RealtimeAllocationGuard {
public:
  RealtimeAllocationGuard() noexcept;
  ~RealtimeAllocationGuard();
  RealtimeAllocationGuard(const RealtimeAllocationGuard&) = delete;
  RealtimeAllocationGuard& operator=(const RealtimeAllocationGuard&) = delete;
  HeapTraffic finish() noexcept;

private:
  HeapTraffic traffic_;
  HeapTraffic* previous_;
};

}  // namespace audio_plugin_test
