#include "StemgenRT/LowBandStabilizer.h"
#include <algorithm>
#include <cmath>

namespace audio_plugin {

void LowBandStabilizer::prepare(double sampleRate) {
    if (sampleRate <= 0.0) {
        alpha_ = 0.0f;
        energyAlpha_ = 0.0f;
        highEnvAlpha_ = 0.0f;
        highGateAlpha_ = 0.0f;
        reset();
        return;
    }

    constexpr float kTwoPi = 6.28318530718f;
    const float w = kTwoPi * kLowBandStabilizerCutoffHz / static_cast<float>(sampleRate);
    alpha_ = 1.0f - std::exp(-w);
    alpha_ = std::clamp(alpha_, 0.0f, 1.0f);

    // Separate steep HP detector for "true high-frequency presence" gating.
    constexpr float kHighSenseCutoffHz = 350.0f;
    constexpr float kButterworthQ = 0.7071067811865476f;
    auto hpCoeffs = juce::dsp::IIR::Coefficients<float>::makeHighPass(
        sampleRate, kHighSenseCutoffHz, kButterworthQ);
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        highSenseHp1_[ch].coefficients = hpCoeffs;
        highSenseHp2_[ch].coefficients = hpCoeffs;
    }

    // Smooth low-band energy so per-stem redistribution weights do not jitter.
    constexpr float kEnergySmoothingHz = 25.0f;
    const float wEnergy = kTwoPi * kEnergySmoothingHz / static_cast<float>(sampleRate);
    energyAlpha_ = 1.0f - std::exp(-wEnergy);
    energyAlpha_ = std::clamp(energyAlpha_, 0.0f, 1.0f);

    // Envelope follower for dry high-band / leakage high-band magnitudes.
    constexpr float kHighEnvSmoothingHz = 50.0f;
    const float wHighEnv = kTwoPi * kHighEnvSmoothingHz / static_cast<float>(sampleRate);
    highEnvAlpha_ = 1.0f - std::exp(-wHighEnv);
    highEnvAlpha_ = std::clamp(highEnvAlpha_, 0.0f, 1.0f);

    // Additional smoothing for high-band leakage gain to avoid zippering.
    constexpr float kHighGateSmoothingHz = 30.0f;
    const float wHighGate = kTwoPi * kHighGateSmoothingHz / static_cast<float>(sampleRate);
    highGateAlpha_ = 1.0f - std::exp(-wHighGate);
    highGateAlpha_ = std::clamp(highGateAlpha_, 0.0f, 1.0f);

    reset();
}

void LowBandStabilizer::reset() {
    for (auto& s : dryLowState_) {
        s = 0.0f;
    }
    for (auto& stem : stemLowState_) {
        for (auto& s : stem) {
            s = 0.0f;
        }
    }
    for (auto& stem : stemEnergyState_) {
        for (auto& s : stem) {
            s = 0.0f;
        }
    }
    for (auto& s : dryHighEnvState_) {
        s = 0.0f;
    }
    for (auto& s : dryLowEnvState_) {
        s = 0.0f;
    }
    for (auto& s : leakHighEnvState_) {
        s = 0.0f;
    }
    for (auto& s : highLeakGainState_) {
        s = 1.0f;
    }
    for (auto& s : drumsLeakHighEnvState_) {
        s = 0.0f;
    }
    for (auto& s : drumsHighLeakGainState_) {
        s = 1.0f;
    }
    for (auto& s : lowDominanceGateState_) {
        s = 1.0f;
    }
    for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
        highSenseHp1_[ch].reset();
        highSenseHp2_[ch].reset();
    }
}

void LowBandStabilizer::processStereo(
    float dryL, float dryR,
    StemPostProcessor::StemSamples& stemsL,
    StemPostProcessor::StemSamples& stemsR) {
    if (alpha_ <= 0.0f) {
        return;
    }
    processChannel(0, dryL, stemsL);
    processChannel(1, dryR, stemsR);
}

float LowBandStabilizer::processOnePole(float input, float& state) const {
    state += alpha_ * (input - state);
    return state;
}

void LowBandStabilizer::processChannel(
    int channel, float drySample, StemPostProcessor::StemSamples& stems) {
    const size_t ch = static_cast<size_t>(channel);
    constexpr float kEpsilon = 1e-8f;
    constexpr float kVocalsLowWeightMax = 0.0f;
    constexpr float kOtherLowWeightMax = 0.0f;
    constexpr float kHighLeakAllowance = 1.20f;
    constexpr float kDrumsHighLeakAllowance = 0.35f;
    constexpr float kLowDominanceRatioFloor = 0.005f;
    constexpr float kLowDominanceRatioCeil = 0.040f;
    constexpr float kDryHighPresenceFloor = 0.0008f;
    constexpr float kDryHighPresenceCeil = 0.0060f;

    float stemVals[kNumStems] = {stems.drums, stems.bass, stems.vocals, stems.other};
    float lowVals[kNumStems] = {0.0f, 0.0f, 0.0f, 0.0f};
    float highVals[kNumStems] = {0.0f, 0.0f, 0.0f, 0.0f};
    float smoothedEnergies[kNumStems] = {0.0f, 0.0f, 0.0f, 0.0f};

    const float dryLow = processOnePole(drySample, dryLowState_[ch]);
    const float dryHighSense =
        highSenseHp2_[ch].processSample(highSenseHp1_[ch].processSample(drySample));

    float energySum = 0.0f;
    for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        lowVals[stem] = processOnePole(stemVals[stem], stemLowState_[stem][ch]);
        highVals[stem] = stemVals[stem] - lowVals[stem];
        const float instEnergy = lowVals[stem] * lowVals[stem];
        auto& energyState = stemEnergyState_[stem][ch];
        energyState += energyAlpha_ * (instEnergy - energyState);
        smoothedEnergies[stem] = std::max(0.0f, energyState);
        energySum += smoothedEnergies[stem];
    }

    // Keep most low-band content in drums/bass to suppress low-frequency buzz
    // leaking into vocals/other, while preserving model-driven balance.
    float wVocals = 0.0f;
    float wOther = 0.0f;
    if (energySum > kEpsilon) {
        wVocals = smoothedEnergies[2] / energySum;
        wOther = smoothedEnergies[3] / energySum;
    }
    wVocals = std::min(wVocals, kVocalsLowWeightMax);
    wOther = std::min(wOther, kOtherLowWeightMax);

    const float remaining = std::max(0.0f, 1.0f - wVocals - wOther);
    const float drumsBassEnergy = smoothedEnergies[0] + smoothedEnergies[1];
    float wDrums = 0.5f * remaining;
    float wBass = 0.5f * remaining;
    if (drumsBassEnergy > kEpsilon) {
        wDrums = remaining * (smoothedEnergies[0] / drumsBassEnergy);
        wBass = remaining * (smoothedEnergies[1] / drumsBassEnergy);
    }

    const float weights[kNumStems] = {wDrums, wBass, wVocals, wOther};

    // Suppress high-band leakage in vocals/other when input has little high-band energy.
    // This directly targets the "buzz" heard in other/vocals on low-passed kick input.
    const float dryHighMag = std::abs(dryHighSense);
    auto& dryHighEnv = dryHighEnvState_[ch];
    dryHighEnv += highEnvAlpha_ * (dryHighMag - dryHighEnv);
    auto& dryLowEnv = dryLowEnvState_[ch];
    dryLowEnv += highEnvAlpha_ * (std::abs(dryLow) - dryLowEnv);

    const float leakHighMag = std::abs(highVals[2]) + std::abs(highVals[3]);
    auto& leakHighEnv = leakHighEnvState_[ch];
    leakHighEnv += highEnvAlpha_ * (leakHighMag - leakHighEnv);

    // Detect low-band-dominant input (e.g., LPF kick). We use this to
    // selectively suppress synthetic high-band buzz in stems.
    const float highToLowRatio = dryHighEnv / (dryLowEnv + kEpsilon);
    const float targetLowDominanceGate = std::clamp(
        (highToLowRatio - kLowDominanceRatioFloor) /
            (kLowDominanceRatioCeil - kLowDominanceRatioFloor),
        0.0f, 1.0f);
    const float targetHighPresenceGate = std::clamp(
        (dryHighEnv - kDryHighPresenceFloor) /
            (kDryHighPresenceCeil - kDryHighPresenceFloor),
        0.0f, 1.0f);
    const float targetStemGate = std::min(targetLowDominanceGate, targetHighPresenceGate);
    const float lowOnlyAmount = 1.0f - targetStemGate;

    float targetLeakGain = 1.0f;
    if (leakHighEnv > kEpsilon) {
        targetLeakGain = std::clamp((dryHighEnv * kHighLeakAllowance) / (leakHighEnv + kEpsilon),
                                    0.0f, 1.0f);
    }
    auto& leakGain = highLeakGainState_[ch];
    leakGain += highGateAlpha_ * (targetLeakGain - leakGain);
    highVals[2] *= leakGain;
    highVals[3] *= leakGain;

    // Drums can still carry model hiss on low-only content. Constrain drums
    // high-band leakage only when input has little/no high-band energy.
    const float drumsLeakHighMag = std::abs(highVals[0]);
    auto& drumsLeakEnv = drumsLeakHighEnvState_[ch];
    drumsLeakEnv += highEnvAlpha_ * (drumsLeakHighMag - drumsLeakEnv);
    float targetDrumsLeakGain = 1.0f;
    if (drumsLeakEnv > kEpsilon) {
        targetDrumsLeakGain =
            std::clamp((dryHighEnv * kDrumsHighLeakAllowance) / (drumsLeakEnv + kEpsilon),
                       0.0f, 1.0f);
    }
    const float targetDrumsGate = 1.0f - lowOnlyAmount * (1.0f - targetDrumsLeakGain);
    auto& drumsLeakGain = drumsHighLeakGainState_[ch];
    drumsLeakGain += highGateAlpha_ * (targetDrumsGate - drumsLeakGain);
    highVals[0] *= drumsLeakGain;

    for (size_t stem = 0; stem < static_cast<size_t>(kNumStems); ++stem) {
        stemVals[stem] = highVals[stem] + dryLow * weights[stem];
    }

    // When input is strongly low-band-dominant (e.g., LPF kick), suppress
    // vocals/other stems to avoid audible synthetic buzz.
    auto& lowDomGate = lowDominanceGateState_[ch];
    lowDomGate += highGateAlpha_ * (targetStemGate - lowDomGate);
    stemVals[2] *= lowDomGate;
    stemVals[3] *= lowDomGate;

    stems.drums = stemVals[0];
    stems.bass = stemVals[1];
    stems.vocals = stemVals[2];
    stems.other = stemVals[3];
}

}  // namespace audio_plugin
