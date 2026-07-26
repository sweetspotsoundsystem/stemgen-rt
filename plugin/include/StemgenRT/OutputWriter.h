#pragma once

#include <array>
#include <cstddef>
#include <vector>
#include "Constants.h"

namespace audio_plugin {

class OverlapAddProcessor;

// Writes the stateful graph output to native plugin buses, fades only
// unreliable near-silence drums/bass/vocals model contributions, crossfades to
// an unchanged latency-aligned fail-safe signal on underruns, and routes the
// final residual to Other so the four stems sum exactly to Main.
class OutputWriter {
public:
  OutputWriter();

  struct WriteResult {
    // Missing exact-timeline model samples at or beyond the configured
    // plugin latency. Pipeline fill before that boundary is intentional
    // startup, not an underrun.
    size_t underrunSamples{0};
    bool hadUnderrun{false};
    bool isUnderrunNow{false};
    // Diagnostics (debug builds only)
    size_t ringAvailAtStart{0};  // Ring buffer samples available at block start
    float crossfadeGainAtStart{0.0f};  // Crossfade gain at block start
    bool underrunTransition{
        false};  // True when transitioning into underrun state
  };

  // Configure host-domain confidence and crossfade timing. The graph remains
  // at 44.1 kHz, while converted stems are written on the host clock.
  void prepare(double sampleRate);

  // Reset crossfade and low-level confidence state (call on transport start)
  void reset();

  // Set up write pointers for a block (call at start of processBlock)
  // mainWrite: pointers to main bus channels [kNumChannels]
  // mainNumCh: number of channels in main bus
  // stemWrite: pointers to stem bus channels [4][kNumChannels]
  // stemNumCh: number of channels per stem bus [4]
  void setOutputPointers(float* mainWrite[kNumChannels],
                         int mainNumCh,
                         float* stemWrite[4][kNumChannels],
                         int stemNumCh[4]);

  // Write output samples for the block
  // Handles low-level model confidence and the separated/dry crossfade.
  // Missing model samples remain dry fallback regardless of telemetry state.
  // Set underrunTelemetryEnabled false when a known host transport is stopped,
  // because idle callbacks have no playback deadline to miss.
  // Set modelOutputEnabled false whenever the actual callback exceeds the
  // prepared PDC reserve. Scheduled samples are discarded on their original
  // timeline and the complete mixture is routed to Other.
  WriteResult writeBlock(
      OverlapAddProcessor& overlapAdd,
      const std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>&
          outputRingBuffers,
      const std::array<std::vector<float>, kNumChannels>& delayedInputBuffer,
      size_t ringSize,
      int numSamples,
      bool underrunTelemetryEnabled = true,
      bool modelOutputEnabled = true,
      bool useScheduledModelReference = true);

  // Get current crossfade gain (for state persistence)
  float getCrossfadeGain() const { return crossfadeGain_; }

private:
  // Output bus pointers (set per block)
  float* mainWrite_[kNumChannels] = {nullptr, nullptr};
  int mainNumCh_ = 0;
  float* stemWrite_[4][kNumChannels] = {{nullptr, nullptr},
                                        {nullptr, nullptr},
                                        {nullptr, nullptr},
                                        {nullptr, nullptr}};
  int stemNumCh_[4] = {0, 0, 0, 0};

  // Crossfade state (persists across blocks)
  float crossfadeGain_{0.0f};  // 1.0 = full separated, 0.0 = full dry
  bool wasUnderrun_{false};    // Track underrun transitions for diagnostics
  int underrunCrossfadeSamples_{kUnderrunCrossfadeSamples};

  // Stereo-linked peak follower used to reject the graph's observed
  // near-silence floor without changing the model input or recurrent state.
  float levelEnvelope_{0.0f};
  float separationConfidence_{0.0f};
  float levelReleaseCoefficient_{0.0f};
  int levelHoldSamples_{0};
  int levelHoldSamplesRemaining_{0};

  float updateSeparationConfidence(float linkedPeak);

  // Model output order: 0=drums, 1=bass, 2=vocals, 3=other
  // Bus order: 1=Drums, 2=Bass, 3=Other, 4=Vocals
  static constexpr size_t kBusToStemMap[4] = {0, 1, 3, 2};
};

}  // namespace audio_plugin
