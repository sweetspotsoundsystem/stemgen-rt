#include "StemgenRT/ResidualProcessor.h"

namespace audio_plugin {

ResidualProcessor::StemSamples ResidualProcessor::process(
    float originalSample,
    const StemSamples& stems,
    float gatedVocals,
    float vocalsToOther,
    float softGateGain) {

    // Adjusted other stem includes transferred vocals content
    float otherAdj = stems.other + vocalsToOther;

    // Compute residual using original stems (before vocals gate adjustment)
    float residual = originalSample - (stems.drums + stems.bass + stems.vocals + stems.other);

    // Power-weighted distribution (excluding vocals to keep it clean)
    float p_drums = stems.drums * stems.drums;
    float p_bass = stems.bass * stems.bass;
    float p_other = otherAdj * otherAdj;
    float totalPower = p_drums + p_bass + p_other + kResidualEpsilon;

    // Distribute residual proportionally to non-vocal stem power
    // Apply soft gate to eliminate model noise floor on quiet passages
    StemSamples result;
    result.drums = (stems.drums + (p_drums / totalPower) * residual) * softGateGain;
    result.bass = (stems.bass + (p_bass / totalPower) * residual) * softGateGain;
    result.vocals = gatedVocals * softGateGain;  // Gated vocals, no residual
    result.other = (otherAdj + (p_other / totalPower) * residual) * softGateGain;

    return result;
}

void ResidualProcessor::processStereo(
    float originalL, float originalR,
    const StemSamples& stemsL, const StemSamples& stemsR,
    float gatedVocalsL, float gatedVocalsR,
    float vocalsToOtherL, float vocalsToOtherR,
    float softGateGain,
    StemSamples& outL, StemSamples& outR) {

    outL = process(originalL, stemsL, gatedVocalsL, vocalsToOtherL, softGateGain);
    outR = process(originalR, stemsR, gatedVocalsR, vocalsToOtherR, softGateGain);
}

}  // namespace audio_plugin
