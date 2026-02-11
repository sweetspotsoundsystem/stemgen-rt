#pragma once

#include <array>
#include "Constants.h"

namespace audio_plugin {

// Applies post-model stem processing:
// - transfers gated vocals energy to "other"
// - applies input-following soft gate to all stems
// No residual redistribution.
class StemPostProcessor {
public:
    // Small epsilon reserved for future numerical guards in post-processing.
    static constexpr float kEpsilon = 1e-10f;

    // Structure to hold per-sample stem values
    struct StemSamples {
        float drums;
        float bass;
        float vocals;
        float other;
    };

    // Process a single-channel stem sample bundle.
    // stems: raw stem outputs from model (after LP reinjection)
    // gatedVocals: vocals after gating (from VocalsGate)
    // vocalsToOther: vocals content transferred to other (from VocalsGate)
    // softGateGain: gate gain to apply to all stems (from SoftGate)
    static StemSamples process(
        const StemSamples& stems,
        float gatedVocals,
        float vocalsToOther,
        float softGateGain);

    // Process stereo sample pair.
    static void processStereo(
        const StemSamples& stemsL, const StemSamples& stemsR,
        float gatedVocalsL, float gatedVocalsR,
        float vocalsToOtherL, float vocalsToOtherR,
        float softGateGain,
        StemSamples& outL, StemSamples& outR);
};

}  // namespace audio_plugin
