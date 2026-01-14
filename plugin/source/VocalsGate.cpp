#include "StemgenRT/VocalsGate.h"

namespace audio_plugin {

void VocalsGate::prepare(double sampleRate) {
    // coeff = exp(-1 / (sampleRate * timeInSeconds))
    attackCoeff_ = std::exp(-1.0f / (static_cast<float>(sampleRate) * kVocalsGateAttackTimeSec));
    releaseCoeff_ = std::exp(-1.0f / (static_cast<float>(sampleRate) * kVocalsGateReleaseTimeSec));
    reset();
}

void VocalsGate::reset() {
    smoothedGain_ = 1.0f;
}

float VocalsGate::calculateTarget(float vocalsEnergy, float totalStemEnergy, float vocalsPeak) {
    // Criterion 1: Ratio of vocals energy to total mix
    float vocalsRatio = vocalsEnergy / totalStemEnergy;
    float gateRatio = std::clamp((vocalsRatio - kVocalsGateRatioFloor) * kVocalsGateRatioInvRange, 0.0f, 1.0f);

    // Criterion 2: Absolute vocals level
    float gateLevel = std::clamp((vocalsPeak - kVocalsGateLevelFloor) * kVocalsGateLevelInvRange, 0.0f, 1.0f);

    // Most restrictive wins
    return std::min(gateRatio, gateLevel);
}

float VocalsGate::process(float vocalsEnergy, float totalStemEnergy, float vocalsPeak) {
    float target = calculateTarget(vocalsEnergy, totalStemEnergy, vocalsPeak);

    // Asymmetric attack/release smoothing
    // Fast attack (gate opening) so vocals come in quickly
    // Slow release (gate closing) to avoid pumping on short gaps
    float coeff = (target > smoothedGain_) ? attackCoeff_ : releaseCoeff_;
    smoothedGain_ = coeff * smoothedGain_ + (1.0f - coeff) * target;

    return smoothedGain_;
}

}  // namespace audio_plugin
