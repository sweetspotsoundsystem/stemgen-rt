#pragma once

#include <algorithm>
#include <cmath>
#include "Constants.h"

namespace audio_plugin {

// Dual-criteria vocals gate with asymmetric attack/release smoothing.
// Detects when vocals energy is very low relative to the mix and provides
// gain attenuation to transfer spurious vocals content to "other" stem.
//
// Two detection criteria (most restrictive wins):
// 1. Ratio-based: vocals are tiny fraction of total mix (likely instrumental)
// 2. Level-based: vocals are very quiet in absolute terms
//
// The gate uses asymmetric smoothing: fast attack (vocals come in quickly),
// slow release (avoid pumping on gaps).
class VocalsGate {
public:
    VocalsGate() = default;

    // Prepare the gate with sample rate (calculates smoothing coefficients)
    void prepare(double sampleRate);

    // Reset the smoothed state
    void reset();

    // Process one sample and return the smoothed gate gain (0.0 to 1.0).
    // - vocalsEnergy: vocals_L^2 + vocals_R^2
    // - totalStemEnergy: sum of all stems' energy (should include small epsilon)
    // - vocalsPeak: max(abs(vocals_L), abs(vocals_R))
    //
    // Returns the smoothed gain to apply to vocals.
    // Caller should transfer (1 - gain) * vocals to "other" stem.
    float process(float vocalsEnergy, float totalStemEnergy, float vocalsPeak);

    // Get the current smoothed gate gain (useful for metering)
    float getSmoothedGain() const { return smoothedGain_; }

private:
    // Calculate the instantaneous (non-smoothed) gate target
    static float calculateTarget(float vocalsEnergy, float totalStemEnergy, float vocalsPeak);

    float smoothedGain_{1.0f};
    float attackCoeff_{0.9985f};   // Default: ~15ms attack at 44.1kHz
    float releaseCoeff_{0.99994f}; // Default: ~400ms release at 44.1kHz
};

}  // namespace audio_plugin
