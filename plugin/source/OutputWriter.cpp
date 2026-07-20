#include "StemgenRT/OutputWriter.h"
#include "StemgenRT/OverlapAddProcessor.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace audio_plugin {

OutputWriter::OutputWriter() {
  prepare(static_cast<double>(kModelSampleRate));
}

void OutputWriter::prepare(double sampleRate) {
  const double safeSampleRate =
      sampleRate > 0.0 ? sampleRate : static_cast<double>(kModelSampleRate);
  // Attack immediately, hold peaks across low-frequency zero crossings, then
  // reach -60 dB after kLowLevelReleaseSeconds of uninterrupted decay.
  levelReleaseCoefficient_ = static_cast<float>(
      std::exp(std::log(0.001) / (static_cast<double>(kLowLevelReleaseSeconds) *
                                  safeSampleRate)));
  levelHoldSamples_ = std::max(
      1, static_cast<int>(std::lround(
             static_cast<double>(kLowLevelHoldSeconds) * safeSampleRate)));
  underrunCrossfadeSamples_ =
      std::max(1, static_cast<int>(std::lround(
                      static_cast<double>(kUnderrunCrossfadeSamples) *
                      safeSampleRate / static_cast<double>(kModelSampleRate))));
  reset();
}

void OutputWriter::reset() {
  crossfadeGain_ = 0.0f;
  wasUnderrun_ = false;
  levelEnvelope_ = 0.0f;
  separationConfidence_ = 0.0f;
  levelHoldSamplesRemaining_ = 0;
}

float OutputWriter::updateSeparationConfidence(float linkedPeak) {
  if (!std::isfinite(linkedPeak)) {
    levelEnvelope_ = 0.0f;
    separationConfidence_ = 0.0f;
    levelHoldSamplesRemaining_ = 0;
    return separationConfidence_;
  }

  if (linkedPeak >= levelEnvelope_) {
    levelEnvelope_ = linkedPeak;
    levelHoldSamplesRemaining_ = levelHoldSamples_;
  } else if (levelHoldSamplesRemaining_ > 0) {
    --levelHoldSamplesRemaining_;
  } else {
    levelEnvelope_ = levelReleaseCoefficient_ * levelEnvelope_ +
                     (1.0f - levelReleaseCoefficient_) * linkedPeak;
  }

  constexpr float inverseRange =
      1.0f / (kLowLevelSeparationOpen - kLowLevelSeparationClosed);
  const float linearConfidence = std::clamp(
      (levelEnvelope_ - kLowLevelSeparationClosed) * inverseRange, 0.0f, 1.0f);
  // Smoothstep avoids a slope discontinuity at either threshold.
  separationConfidence_ =
      linearConfidence * linearConfidence * (3.0f - 2.0f * linearConfidence);
  return separationConfidence_;
}

void OutputWriter::setOutputPointers(float* mainWrite[kNumChannels],
                                     int mainNumCh,
                                     float* stemWrite[4][kNumChannels],
                                     int stemNumCh[4]) {
  for (int ch = 0; ch < kNumChannels; ++ch) {
    mainWrite_[ch] = mainWrite[ch];
  }
  mainNumCh_ = mainNumCh;

  for (int b = 0; b < 4; ++b) {
    for (int ch = 0; ch < kNumChannels; ++ch) {
      stemWrite_[b][ch] = stemWrite[b][ch];
    }
    stemNumCh_[b] = stemNumCh[b];
  }
}

OutputWriter::WriteResult OutputWriter::writeBlock(
    OverlapAddProcessor& overlapAdd,
    const std::array<std::array<std::vector<float>, kNumChannels>, kNumStems>&
        outputRingBuffers,
    const std::array<std::vector<float>, kNumChannels>& delayedInputBuffer,
    size_t ringSize,
    int numSamples,
    bool underrunTelemetryEnabled,
    bool modelOutputEnabled,
    bool useScheduledModelReference) {
  WriteResult result;
  static_cast<void>(ringSize);

  // Capture diagnostic state at block start
  result.ringAvailAtStart = overlapAdd.getOutputSamplesAvailable();
  result.crossfadeGainAtStart = crossfadeGain_;

  // Local crossfade gain for smooth transitions
  float xfadeGain = crossfadeGain_;
  const float xfadeDelta = 1.0f / static_cast<float>(underrunCrossfadeSamples_);

  for (int i = 0; i < numSamples; ++i) {
    const uint64_t outputTimelineSample = overlapAdd.getOutputTimelineSample();
    const bool underrunTelemetryEligible =
        underrunTelemetryEnabled &&
        outputTimelineSample >=
            static_cast<uint64_t>(overlapAdd.getLatencySamples());
    const bool haveScheduledModel =
        modelOutputEnabled && overlapAdd.hasModelOutputForCurrentSample();
    const size_t readPos = overlapAdd.getOutputReadPos();
    bool scheduledReferenceIsFinite = true;
    bool scheduledRetainedStemsAreFinite = true;
    if (haveScheduledModel) {
      for (int ch = 0; ch < kNumChannels; ++ch) {
        if (useScheduledModelReference) {
          scheduledReferenceIsFinite =
              scheduledReferenceIsFinite &&
              std::isfinite(
                  delayedInputBuffer[static_cast<size_t>(ch)][readPos]);
        }
        for (const int stem : {kStemDrums, kStemBass, kStemVocals}) {
          scheduledRetainedStemsAreFinite =
              scheduledRetainedStemsAreFinite &&
              std::isfinite(outputRingBuffers[static_cast<size_t>(
                  stem)][static_cast<size_t>(ch)][readPos]);
        }
      }
    }
    const bool haveAlignedReference =
        haveScheduledModel && scheduledReferenceIsFinite;
    const bool have = haveAlignedReference && scheduledRetainedStemsAreFinite;
    if (!have && underrunTelemetryEligible) {
      result.hadUnderrun = true;
      ++result.underrunSamples;
    }

    // A missing model sample has no same-timeline separated value to fade
    // from, so switch immediately to the complete dry split. When exact-
    // timeline output resumes, fade it in against that same dry sample.
    if (have) {
      xfadeGain = std::min(1.0f, xfadeGain + xfadeDelta);
    } else {
      xfadeGain = 0.0f;
    }

    // Get dry signal (latency-aligned input) for fallback/crossfade
    float dry[kNumChannels];
    for (int ch = 0; ch < kNumChannels; ++ch) {
      const float sample = overlapAdd.readDryDelaySample(ch);
      dry[ch] = std::isfinite(sample) ? sample : 0.0f;
    }

    // Main is always the complete latency-aligned mixture. A scheduled raw
    // reference and the dry delay represent the same input timeline, but
    // selecting rather than crossfading prevents an underrun from fading
    // Main toward zero when no model sample exists.
    float mainOutput[kNumChannels];
    float levelReference[kNumChannels];
    for (int ch = 0; ch < kNumChannels; ++ch) {
      const float scheduledSample =
          delayedInputBuffer[static_cast<size_t>(ch)][readPos];
      mainOutput[ch] = useScheduledModelReference && haveAlignedReference
                           ? scheduledSample
                           : dry[ch];
      levelReference[ch] = mainOutput[ch];
      if (ch < mainNumCh_ && mainWrite_[ch] != nullptr) {
        mainWrite_[ch][i] = mainOutput[ch];
      }
    }

    const bool levelReferenceIsFinite =
        (!useScheduledModelReference || scheduledReferenceIsFinite) &&
        std::isfinite(levelReference[0]) && std::isfinite(levelReference[1]);
    const float linkedPeak =
        levelReferenceIsFinite
            ? std::max(std::abs(levelReference[0]), std::abs(levelReference[1]))
            : std::numeric_limits<float>::quiet_NaN();
    const float separationConfidence = updateSeparationConfidence(linkedPeak);
    const float modelGain = xfadeGain * separationConfidence;

    // Fade only the three retained model sources. During startup, an underrun,
    // or a non-finite result, the complete latency-aligned mixture is routed to
    // Other instead of leaking one quarter of it into every named source. The
    // final residual preserves Main == Drums + Bass + Vocals + Other through
    // provider differences and every fallback transition.
    for (int ch = 0; ch < kNumChannels; ++ch) {
      float stems[kNumStems] = {};
      for (const int stem : {kStemDrums, kStemBass, kStemVocals}) {
        const float separated =
            have ? outputRingBuffers[static_cast<size_t>(stem)]
                                    [static_cast<size_t>(ch)][readPos]
                 : 0.0f;
        stems[stem] = modelGain * separated;
      }
      stems[kStemOther] = mainOutput[ch] - stems[kStemDrums] -
                          stems[kStemBass] - stems[kStemVocals];

      // This is a final defense after the runtime's tensor validation. If an
      // extreme but finite provider value overflows during the residual, keep
      // the output finite and mixture-lossless by selecting complete fallback
      // for this channel/sample.
      if (!std::isfinite(stems[kStemDrums]) ||
          !std::isfinite(stems[kStemBass]) ||
          !std::isfinite(stems[kStemVocals]) ||
          !std::isfinite(stems[kStemOther])) {
        stems[kStemDrums] = 0.0f;
        stems[kStemBass] = 0.0f;
        stems[kStemVocals] = 0.0f;
        stems[kStemOther] = mainOutput[ch];
      }

      for (int busIdx = 0; busIdx < 4; ++busIdx) {
        if (ch >= stemNumCh_[busIdx] || stemWrite_[busIdx][ch] == nullptr) {
          continue;
        }
        stemWrite_[busIdx][ch][i] = stems[kBusToStemMap[busIdx]];
      }
    }

    // Both cursors advance on every rendered sample. Scheduled model data
    // is tagged with this exact timeline; late results can never shift it.
    overlapAdd.advanceOutputTimeline();
    overlapAdd.advanceDryDelayPos();
  }
  crossfadeGain_ = xfadeGain;

  // UI telemetry follows exact missing post-latency samples. The initial
  // fade-in before/at the PDC boundary is expected pipeline startup.
  result.isUnderrunNow = result.hadUnderrun;

  // Detect underrun transition (entering underrun state)
  result.underrunTransition = (result.hadUnderrun && !wasUnderrun_);
  wasUnderrun_ = result.hadUnderrun;

  return result;
}

}  // namespace audio_plugin
