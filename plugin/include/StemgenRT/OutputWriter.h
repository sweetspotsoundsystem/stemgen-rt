#pragma once

#include <array>
#include <cstddef>
#include <vector>
#include "Constants.h"

namespace audio_plugin {

class OverlapAddProcessor;

// Handles writing separated stems to output buses with crossfade support.
// Manages the crossfade between separated audio and dry fallback signal
// during underrun conditions.
class OutputWriter {
public:
    OutputWriter() = default;

    // Reset crossfade state (call on transport start)
    void reset();

    // Set up write pointers for a block (call at start of processBlock)
    // mainWrite: pointers to main bus channels [kNumChannels]
    // mainNumCh: number of channels in main bus
    // stemWrite: pointers to stem bus channels [4][kNumChannels]
    // stemNumCh: number of channels per stem bus [4]
    void setOutputPointers(float* mainWrite[kNumChannels], int mainNumCh,
                           float* stemWrite[4][kNumChannels], int stemNumCh[4]);

    // Write output samples for the block
    // Handles crossfade between separated and dry signal
    // Returns the number of separated samples consumed from ring buffer
    void writeBlock(
        OverlapAddProcessor& overlapAdd,
        const std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>& outputRingBuffers,
        size_t ringSize,
        int numSamples);

    // Get current crossfade gain (for state persistence)
    float getCrossfadeGain() const { return crossfadeGain_; }

private:
    // Output bus pointers (set per block)
    float* mainWrite_[kNumChannels] = {nullptr, nullptr};
    int mainNumCh_ = 0;
    float* stemWrite_[4][kNumChannels] = {{nullptr, nullptr}, {nullptr, nullptr},
                                          {nullptr, nullptr}, {nullptr, nullptr}};
    int stemNumCh_[4] = {0, 0, 0, 0};

    // Crossfade state (persists across blocks)
    float crossfadeGain_{1.0f};  // 1.0 = full separated, 0.0 = full dry

    // Model output order: 0=drums, 1=bass, 2=vocals, 3=other
    // Bus order: 1=Drums, 2=Bass, 3=Other, 4=Vocals
    static constexpr size_t kBusToStemMap[4] = {0, 1, 3, 2};
};

}  // namespace audio_plugin
