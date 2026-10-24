#pragma once
#include <array>
#include <memory>
#include <juce_core/juce_core.h>
#include "WorkerWorkgroup.h"

struct OrtApi;
struct OrtEnv;
struct OrtSessionOptions;
struct OrtMemoryInfo;
struct OrtValue;

namespace audio_plugin {

// Experimental: the calling inference worker runs Bass; two persistent helpers
// run Drums and Vocals. Only the calling worker dispatches/joins a complete
// hop. No audio-thread code touches these sessions or waits for helper
// completion.
class ParallelInference {
public:
  ParallelInference();
  ~ParallelInference();
  ParallelInference(const ParallelInference&) = delete;
  ParallelInference& operator=(const ParallelInference&) = delete;
  bool load(const OrtApi*,
            OrtEnv*,
            const juce::File& directory,
            juce::String& error);
  bool prepare(OrtMemoryInfo*,
               const std::array<OrtValue*, 14>& inputs,
               const std::array<OrtValue*, 14>& outputs,
               float* separated,
               juce::String& error);
  bool start(bool realtime, std::array<WorkerWorkgroup*, 2> workgroups);
  void stop();
  bool run();
  // 0=not attempted, 1=applied, 2=unsupported, 3=failed. Readback is not timing
  // qualification.
  int initialPolicy(size_t helper) const noexcept;
  int finalPolicy(size_t helper) const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace audio_plugin
