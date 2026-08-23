#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "Constants.h"

namespace audio_plugin {

// Owns the audio-thread accumulation, output-ring, and latency-aligned dry
// fallback buffers. Model overlap-add now lives inside the ONNX graph; the
// historical class name is retained to avoid unnecessary API churn.
class OverlapAddProcessor {
public:
  OverlapAddProcessor();

  void allocate(
      size_t maximumHostBlockSize = kOutputChunkSize,
      size_t latencySamples = kPluginLatencySamples,
      size_t maximumModelOutputSamples = kOutputChunkSize);
  void reset();
  // Audio-thread reset. Invalidate buffered model/dry data by generation and
  // counters; do not clear the backing storage here.
  void resetIndices();
  // Logically invalidate dry history without touching its backing storage.
  // reset() performs the physical clear on the non-real-time path.
  void clearDryDelayBuffer();

  size_t getInputAccumCount() const { return inputAccumCount_; }
  // The host-rate dry timeline and the fixed 44.1 kHz model accumulator are
  // separate clocks when sample-rate conversion is active. The legacy
  // pushInputSample() keeps the exact 44.1 kHz one-call path intact.
  void pushDryInputSample(int channel, float sample);
  void pushModelInputSample(int channel, float sample);
  void pushInputSample(int channel, float sample);
  bool readyForInference() const {
    return inputAccumCount_ >= static_cast<size_t>(kOutputChunkSize);
  }
  const std::array<std::vector<float>, kNumChannels>& getInputAccumBuffer()
      const {
    return inputAccumBuffer_;
  }
  void clearInputAccum();

  std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>&
  getOutputRingBuffers() {
    return outputRingBuffers_;
  }
  std::array<std::vector<float>, kNumChannels>& getDelayedInputBuffer() {
    return delayedInputBuffer_;
  }

  uint64_t getOutputTimelineSample() const { return outputTimelineSample_; }
  size_t getOutputReadPos() const {
    return static_cast<size_t>(outputTimelineSample_ % getOutputRingSize());
  }
  size_t getOutputRingPosition(uint64_t timelineSample) const {
    return static_cast<size_t>(timelineSample % getOutputRingSize());
  }
  size_t getOutputSamplesAvailable() const { return scheduledOutputSamples_; }
  size_t getOutputRingSize() const { return outputRingBuffers_[0][0].size(); }

  bool canScheduleModelOutput(uint64_t firstTimelineSample,
                              size_t sampleCount) const;
  void markModelOutputScheduled(uint64_t firstTimelineSample,
                                size_t sampleCount);
  bool hasModelOutputForCurrentSample() const;
  void advanceOutputTimeline();

  float readDryDelaySample(int channel) const;
  void advanceDryDelayPos();

  bool canProcessHostBlock(size_t sampleCount) const {
    return sampleCount <= dryDelayHostBlockCapacity_;
  }
  size_t getLatencySamples() const { return latencySamples_; }

private:
  friend class OverlapAddProcessorTestPeer;

  std::array<std::vector<float>, kNumChannels> inputAccumBuffer_;
  size_t inputAccumCount_{0};

  std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>
      outputRingBuffers_;
  std::array<std::vector<float>, kNumChannels> delayedInputBuffer_;
  std::vector<uint64_t> outputTimelineTags_;
  std::vector<uint64_t> outputGenerationTags_;
  uint64_t outputGeneration_{0};
  uint64_t outputTimelineSample_{0};
  size_t scheduledOutputSamples_{0};

  std::array<std::vector<float>, kNumChannels> dryDelayLine_;
  size_t dryDelayWritePos_{0};
  size_t dryDelayReadPos_{0};
  size_t dryDelayHostBlockCapacity_{0};
  size_t latencySamples_{kPluginLatencySamples};
  uint64_t dryInputSamplesWritten_{0};
  uint64_t dryOutputSamplesRead_{0};

  void clearDryDelayStorage();

  static constexpr uint64_t kInvalidOutputTimelineTag =
      std::numeric_limits<uint64_t>::max();
};

}  // namespace audio_plugin
