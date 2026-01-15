#pragma once

#include <algorithm>
#include <cmath>
#include "Constants.h"

namespace audio_plugin {

// Input-following soft gate for eliminating model noise floor on quiet passages.
// Neural networks output small values even for silent input; this gate attenuates
// stem outputs when input is very quiet, eliminating audible noise when soloing
// stems on silent passages.
//
// This is a stateless utility - it computes gate gain from input magnitude.
// The gain ramps from 0 (at floor) to 1 (at threshold).
class SoftGate {
public:
    // Calculate gate gain from input magnitude.
    // Returns 1.0 above threshold, ramps to 0.0 below floor.
    // inputMag should be the peak input level (e.g., max(abs(L), abs(R)))
    static float calculateGain(float inputMag) {
        return std::clamp((inputMag - kSoftGateFloor) * kSoftGateInvRange, 0.0f, 1.0f);
    }

    // Calculate input magnitude from stereo sample pair
    static float inputMagnitude(float left, float right) {
        return std::max(std::abs(left), std::abs(right));
    }

    // Combined: calculate gate gain from stereo sample pair
    static float calculateGain(float left, float right) {
        return calculateGain(inputMagnitude(left, right));
    }
};

}  // namespace audio_plugin
