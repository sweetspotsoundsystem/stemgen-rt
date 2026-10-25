#pragma once

#include <memory>

namespace audio_plugin_test {

// The standalone host has no Core Audio device keeping its process active.
// Acquire/release only outside measurement; never use this in processBlock.
class MacBenchmarkActivity {
 public:
#if defined(__APPLE__)
  MacBenchmarkActivity();
  ~MacBenchmarkActivity();
  bool isActive() const noexcept;
#else
  MacBenchmarkActivity() = default;
  bool isActive() const noexcept { return false; }
#endif
  MacBenchmarkActivity(const MacBenchmarkActivity&) = delete;
  MacBenchmarkActivity& operator=(const MacBenchmarkActivity&) = delete;

 private:
#if defined(__APPLE__)
  struct Impl;
  std::unique_ptr<Impl> impl_;
#endif
};

}  // namespace audio_plugin_test
