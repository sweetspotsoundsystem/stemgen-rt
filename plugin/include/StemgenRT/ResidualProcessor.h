#pragma once

#include <array>
#include "Constants.h"

namespace audio_plugin {

// Processes residual calculation and power-weighted distribution.
// Ensures sum(stems) = original input for lossless reconstruction.
//
// The residual is distributed to drums/bass/other proportionally to their power,
// while vocals receives no residual (keeps them clean, avoids bleed artifacts).
class ResidualProcessor {
public:
    // Small epsilon to avoid division by zero in power calculations
    static constexpr float kResidualEpsilon = 1e-10f;

    // Structure to hold per-sample stem values
    struct StemSamples {
        float drums;
        float bass;
        float vocals;
        float other;
    };

    // Calculate residual and distribute to non-vocal stems.
    // originalSample: the input sample (fullband)
    // stems: raw stem outputs from model
    // gatedVocals: vocals after gating (from VocalsGate)
    // vocalsToOther: vocals content transferred to other (from VocalsGate)
    // softGateGain: gate gain to apply to all stems (from SoftGate)
    //
    // Returns the adjusted stem values with residual distributed.
    static StemSamples process(
        float originalSample,
        const StemSamples& stems,
        float gatedVocals,
        float vocalsToOther,
        float softGateGain);

    // Process stereo sample pair
    // Returns adjusted stems for both channels
    static void processStereo(
        float originalL, float originalR,
        const StemSamples& stemsL, const StemSamples& stemsR,
        float gatedVocalsL, float gatedVocalsR,
        float vocalsToOtherL, float vocalsToOtherR,
        float softGateGain,
        StemSamples& outL, StemSamples& outR);
};

}  // namespace audio_plugin
