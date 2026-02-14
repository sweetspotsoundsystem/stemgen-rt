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
#include "Crossover.h"
#include "InferenceQueue.h"
#include "InputNormalizer.h"
#include "OnnxRuntime.h"
#include "OutputWriter.h"
#include "OverlapAddProcessor.h"
#include "SoftGate.h"
#include "StemPostProcessor.h"
#include "VocalsGate.h"
#include "LowBandStabilizer.h"

namespace audio_plugin {

class AudioPluginAudioProcessor : public juce::AudioProcessor {
public:
  AudioPluginAudioProcessor();
  ~AudioPluginAudioProcessor() override;

  // Returns a short user-facing status string for ONNX Runtime availability
  // and initialization state, suitable for the editor to display.
  juce::String getOrtStatusString() const;

  // Returns the current plugin latency in samples.
  // This accounts for input accumulation, inference queue depth, and output buffering.
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
  // ONNX Runtime wrapper (handles environment, session, GPU providers, and inference)
  std::unique_ptr<OnnxRuntime> onnxRuntime_;
  juce::String modelLoadError_;  // Stores the last model loading error for display

  // Overlap-add processor (manages all streaming buffers)
  OverlapAddProcessor overlapAdd_;

  // Output writer (handles crossfade between separated and dry signal)
  OutputWriter outputWriter_;

  // Vocals gate with smoothing
  VocalsGate vocalsGate_;

  // Stabilizes low-frequency stem content using dry-signal-constrained redistribution.
  LowBandStabilizer lowBandStabilizer_;

  // Background inference queue (handles thread, requests, and epoch tracking)
  InferenceQueue inferenceQueue_;

  // Chunk sequence tracking for contiguous-only boundary crossfades.
  uint64_t nextInputChunkSequence_{0};
  uint64_t lastOutputChunkSequence_{0};
  bool hasLastOutputChunkSequence_{false};

  // LR4 crossover for low-frequency bypass (splits input into LP + HP)
  Crossover crossover_;

  // Internal methods
  void allocateStreamingBuffers();  // Allocate streaming buffers
  void resetStreamingBuffersRT();  // RT-safe reset (O(1), no memory operations)
#endif

  // Track playback state for hidden state reset
  std::atomic<bool> wasPlaying{false};

  std::atomic<size_t> lastUnderrunSamplesInLastBlock_{0};
  std::atomic<uint64_t> totalUnderrunSamples_{0};
  std::atomic<uint64_t> totalUnderrunBlocks_{0};
  std::atomic<bool> underrunActive_{false};
  std::atomic<uint64_t> outputChunksConsumed_{0};  // Grace period: don't count startup underruns
  std::atomic<size_t> ringFillLevel_{0};           // Ring buffer fill level snapshot

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioPluginAudioProcessor)
};
}  // namespace audio_plugin
