#include "StemgenRT/ModelInputNormalizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace audio_plugin {

void ModelInputNormalizer::allocate() {
  rawPastAudio_.resize(static_cast<size_t>(kNumChannels * kOutputChunkSize));
  reset();
}

void ModelInputNormalizer::reset() {
  std::fill(rawPastAudio_.begin(), rawPastAudio_.end(), 0.0f);
  modelInputGain_ = 1.0f;
  hasPastAudio_ = false;
}

bool ModelInputNormalizer::prepare(
    const std::array<std::vector<float>, kNumChannels>& inputChunk,
    std::vector<float>& normalizedCurrent,
    std::vector<float>& normalizedPast,
    std::array<std::vector<float>, kNumChannels>& alignedInput,
    float& normalizationGain) {
  const size_t hopElements =
      static_cast<size_t>(kNumChannels * kOutputChunkSize);
  if (rawPastAudio_.size() != hopElements ||
      normalizedCurrent.size() != hopElements ||
      normalizedPast.size() != hopElements) {
    return false;
  }

  double currentSumSquares = 0.0;
  double windowSumSquares = 0.0;
  double windowPeak = 0.0;
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    if (inputChunk[ch].size() != static_cast<size_t>(kOutputChunkSize) ||
        alignedInput[ch].size() != static_cast<size_t>(kOutputChunkSize)) {
      return false;
    }

    const size_t channelOffset = ch * static_cast<size_t>(kOutputChunkSize);
    for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
      const float sample = inputChunk[ch][i];
      if (!std::isfinite(sample)) {
        return false;
      }
      const double value = static_cast<double>(sample);
      currentSumSquares += value * value;
      windowPeak = std::max(windowPeak, std::abs(value));
    }

    if (hasPastAudio_) {
      for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
        const float sample = rawPastAudio_[channelOffset + i];
        if (!std::isfinite(sample)) {
          return false;
        }
        const double value = static_cast<double>(sample);
        windowSumSquares += value * value;
        windowPeak = std::max(windowPeak, std::abs(value));
      }
    }
  }
  windowSumSquares += currentSumSquares;

  // Hold the gain through exact digital silence. The current-chunk graph has
  // no flush hop, but retaining the established gain avoids gratuitous state-
  // domain changes across ordinary silent regions.
  if (std::fpclassify(currentSumSquares) == FP_ZERO) {
    normalizationGain = hasPastAudio_ ? modelInputGain_ : 1.0f;
  } else {
    const size_t windowElements = hasPastAudio_ ? 2 * hopElements : hopElements;
    const double rms =
        std::sqrt(windowSumSquares / static_cast<double>(windowElements));
    if (!std::isfinite(rms) || rms <= 0.0) {
      return false;
    }
    const double candidate = static_cast<double>(kModelInputTargetRms) / rms;
    // A low-RMS, high-crest-factor hop (for example, a full-scale impulse)
    // must not be boosted far beyond the model's ordinary amplitude range.
    // Cap added gain against the linked peak over the same raw analysis window
    // used by the RMS detector. max(1, ...) retains the boost-only policy for
    // input that was already above the ceiling.
    const double peakHeadroomGain =
        std::max(1.0, static_cast<double>(kModelInputPeakCeiling) / windowPeak);
    normalizationGain = static_cast<float>(
        std::clamp(std::min(candidate, peakHeadroomGain), 1.0,
                   static_cast<double>(kModelInputMaxBoost)));
  }

  if (!std::isfinite(normalizationGain) || normalizationGain < 1.0f ||
      !std::isfinite(modelInputGain_) || modelInputGain_ < 1.0f) {
    return false;
  }

  if (hasPastAudio_) {
    const float stateScale = normalizationGain / modelInputGain_;
    if (!std::isfinite(stateScale) || stateScale <= 0.0f) {
      return false;
    }
    for (float& sample : normalizedPast) {
      sample *= stateScale;
      if (!std::isfinite(sample)) {
        return false;
      }
    }
  }

  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    const size_t channelOffset = ch * static_cast<size_t>(kOutputChunkSize);
    for (size_t i = 0; i < static_cast<size_t>(kOutputChunkSize); ++i) {
      const size_t offset = channelOffset + i;
      const float normalized = inputChunk[ch][i] * normalizationGain;
      if (!std::isfinite(normalized)) {
        return false;
      }
      normalizedCurrent[offset] = normalized;
      alignedInput[ch][i] = inputChunk[ch][i];
    }
  }
  return true;
}

void ModelInputNormalizer::commit(
    const std::array<std::vector<float>, kNumChannels>& inputChunk,
    float normalizationGain) {
  for (size_t ch = 0; ch < static_cast<size_t>(kNumChannels); ++ch) {
    std::memcpy(
        rawPastAudio_.data() + ch * static_cast<size_t>(kOutputChunkSize),
        inputChunk[ch].data(),
        static_cast<size_t>(kOutputChunkSize) * sizeof(float));
  }
  modelInputGain_ = normalizationGain;
  hasPastAudio_ = true;
}

}  // namespace audio_plugin
