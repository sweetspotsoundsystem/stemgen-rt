#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <memory>
#include <mutex>
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
#include "StreamingSampleRateAdapter.h"

namespace audio_plugin {

class AudioPluginAudioProcessor : public juce::AudioProcessor {
public:
  AudioPluginAudioProcessor();
  ~AudioPluginAudioProcessor() override;

  // Returns a short user-facing status string for ONNX Runtime availability
  // and initialization state, suitable for the editor to display.
  juce::String getOrtStatusString() const;

  // Returns the current plugin latency in samples.
  // The c126 listening configuration accounts for one asynchronous
  // collection hop; the graph itself emits the current input hop.
  int getLatencySamples() const;

  // Returns the current plugin latency in milliseconds based on sample rate.
  double getLatencyMs() const;

  // Underrun debug telemetry exposed for the editor overlay.
  size_t getUnderrunSamplesInLastBlock() const;
  uint64_t getUnderrunSampleCount() const;
  uint64_t getUnderrunBlockCount() const;
  bool isUnderrunActive() const;

  // Number of exact-timeline model samples still scheduled after the most
  // recent callback. Timeline gaps are deliberately not counted as fill.
  size_t getRingFillLevel() const;

  // Debug telemetry for dropped model output.
  uint64_t getRingOverflowEventCount() const;
  uint64_t getRingOverflowSampleDropCount() const;
  uint64_t getQueueFullChunkDropCount() const;

  // Host-callback timing diagnostics. The active PDC is fixed during
  // prepareToPlay(); these lock-free snapshots report a real-time callback
  // whose size would require a larger scheduling reserve without mutating the
  // stream from the audio thread.
  int getPreparedHostBlockSize() const;
  int getLastHostBlockSize() const;
  int getRequiredLatencySamplesForLastHostBlock() const;
  bool isRealtimeCallbackTimingUnsafe() const;
  uint64_t getUnsafeRealtimeCallbackCount() const;
  InferenceQueue::WorkerPriorityStatus getInferenceWorkerPriorityStatus() const;

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
  juce::String
      modelLoadError_;  // Stores the last model loading error for display
  mutable std::mutex statusMutex_;
  std::atomic<bool> sampleRateSupported_{false};
  std::atomic<int> activeLatencySamples_{0};

  // Streaming buffer owner. Overlap-add itself is inside the ONNX graph.
  OverlapAddProcessor overlapAdd_;

  // Host-rate input remains native for Main/fallback. A separate streaming
  // bridge supplies the graph's fixed 44.1 kHz clock.
  StreamingSampleRateAdapter inputSampleRateAdapter_;
  std::array<std::vector<float>, kNumChannels> sanitizedHostInputScratch_;
  std::array<std::vector<float>, kNumChannels> modelInputScratch_;
  int hostSampleRate_{kModelSampleRate};
  int sampleRateConversionDelaySamples_{0};
  int modelSchedulingLatencySamples_{kPluginLatencySamples};
  bool sampleRateConversionActive_{false};

  // Output writer (handles confidence state, aligned fallback, and exact
  // residual)
  OutputWriter outputWriter_;

  // Background inference queue (handles thread, requests, and epoch tracking)
  InferenceQueue inferenceQueue_;

  // Monotonic input sequence lets the worker detect dropped chunks and reset
  // recurrent model state instead of bridging a discontinuity.
  uint64_t nextInputChunkSequence_{0};

  // Internal methods
  void allocateStreamingBuffers(int maximumHostBlockSize,
                                double hostSampleRate);
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
  std::atomic<size_t> ringFillLevel_{0};  // Ring buffer fill level snapshot
  std::atomic<uint64_t> totalRingOverflowEvents_{0};
  std::atomic<uint64_t> totalRingOverflowSamplesDropped_{0};
  std::atomic<uint64_t> totalQueueFullChunkDrops_{0};
  std::atomic<int> preparedHostBlockSize_{0};
  std::atomic<int> lastHostBlockSize_{0};
  std::atomic<int> requiredLatencySamplesForLastHostBlock_{0};
  std::atomic<bool> realtimeCallbackTimingUnsafe_{false};
  std::atomic<uint64_t> unsafeRealtimeCallbackCount_{0};

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioPluginAudioProcessor)
};
}  // namespace audio_plugin
