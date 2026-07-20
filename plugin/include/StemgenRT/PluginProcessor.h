#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <atomic>
#include <cstdint>
#include "Constants.h"
#include "InferenceQueue.h"
#include "OnnxRuntime.h"
#include "OutputWriter.h"
#include "OverlapAddProcessor.h"

namespace audio_plugin {

class AudioPluginAudioProcessor : public juce::AudioProcessor {
public:
  AudioPluginAudioProcessor();
  ~AudioPluginAudioProcessor() override;

  // Returns a short user-facing status string for ONNX Runtime availability
  // and initialization state, suitable for the editor to display.
  juce::String getOrtStatusString() const;

  // Returns the current plugin latency in samples.
  // This accounts for one asynchronous collection hop and the graph's
  // previous-hop output alignment.
  int getLatencySamples() const;
  
  // Returns the current plugin latency in milliseconds based on sample rate.
  double getLatencyMs() const;

  // Underrun debug telemetry exposed for the editor overlay.
  size_t getUnderrunSamplesInLastBlock() const;
  uint64_t getUnderrunSampleCount() const;
  uint64_t getUnderrunBlockCount() const;
  bool isUnderrunActive() const;

  // Ring buffer fill level (samples available for reading).
  // Reflects the actual pipeline depth beyond the reported PDC latency.
  size_t getRingFillLevel() const;

  // Debug telemetry for dropped model output.
  uint64_t getRingOverflowEventCount() const;
  uint64_t getRingOverflowSampleDropCount() const;
  uint64_t getQueueFullChunkDropCount() const;

  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  void releaseResources() override;

  bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

  void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
  using AudioProcessor::processBlock;

  juce::AudioProcessorEditor* createEditor() override;
  bool hasEditor() const override;

  const juce::String getName() const override;

  bool acceptsMidi() const override;
  bool producesMidi() const override;
  bool isMidiEffect() const override;
  double getTailLengthSeconds() const override;

  int getNumPrograms() override;
  int getCurrentProgram() override;
  void setCurrentProgram(int index) override;
  const juce::String getProgramName(int index) override;
  void changeProgramName(int index, const juce::String& newName) override;

  void getStateInformation(juce::MemoryBlock& destData) override;
  void setStateInformation(const void* data, int sizeInBytes) override;

  // Reset streaming buffers to zeros (call when playback stops/restarts)
  void resetStreamingBuffers();

private:
#if defined(STEMGENRT_USE_ONNXRUNTIME) && STEMGENRT_USE_ONNXRUNTIME
  // ONNX Runtime wrapper (handles the CPU session and persistent graph state)
  std::unique_ptr<OnnxRuntime> onnxRuntime_;
  juce::String modelLoadError_;  // Stores the last model loading error for display
  bool sampleRateSupported_{false};

  // Streaming buffer owner. Overlap-add itself is inside the ONNX graph.
  OverlapAddProcessor overlapAdd_;

  // Output writer (handles latency-aligned fallback and exact residual routing)
  OutputWriter outputWriter_;

  // Background inference queue (handles thread, requests, and epoch tracking)
  InferenceQueue inferenceQueue_;

  // Monotonic input sequence lets the worker detect dropped chunks and reset
  // recurrent model state instead of bridging a discontinuity.
  uint64_t nextInputChunkSequence_{0};

  // Internal methods
  void allocateStreamingBuffers(int maximumHostBlockSize);
  void resetStreamingBuffersRT();
#endif

  // Track playback state for hidden state reset
  std::atomic<bool> wasPlaying{false};
  bool hasExpectedPlayheadPosition_{false};
  int64_t expectedPlayheadPosition_{0};

  std::atomic<size_t> lastUnderrunSamplesInLastBlock_{0};
  std::atomic<uint64_t> totalUnderrunSamples_{0};
  std::atomic<uint64_t> totalUnderrunBlocks_{0};
  std::atomic<bool> underrunActive_{false};
  std::atomic<uint64_t> outputChunksConsumed_{0};  // Grace period: don't count startup underruns
  std::atomic<size_t> ringFillLevel_{0};           // Ring buffer fill level snapshot
  std::atomic<uint64_t> totalRingOverflowEvents_{0};
  std::atomic<uint64_t> totalRingOverflowSamplesDropped_{0};
  std::atomic<uint64_t> totalQueueFullChunkDrops_{0};

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioPluginAudioProcessor)
};
}  // namespace audio_plugin
