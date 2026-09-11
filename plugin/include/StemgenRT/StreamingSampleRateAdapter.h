#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "Constants.h"

namespace audio_plugin {

// Stateful, band-limited sample-rate conversion between the model's 44.1 kHz
// clock and a fixed host clock. prepare() is a control-path operation; after it
// succeeds, process(), reset(), and resetAt() do not allocate.
//
// The non-bypass path is a causal linear-phase windowed-sinc/polyphase FIR.
// Output sample m is evaluated at the exact source position
//
//   m * inputRate / outputRate
//
// and carries filterGroupDelayInputSamples() of content delay. resetAt() makes
// that phase relation explicit for discontinuities and non-zero timelines.
// Samples before a reset origin are zero padded. At 44.1 -> 44.1 kHz the FIR is
// bypassed and finite samples are copied bit-for-bit with zero added delay.
class StreamingSampleRateAdapter final {
public:
  static constexpr std::uint32_t kMaximumSampleRate = 384000U;
  static constexpr std::size_t kMaximumChannelCount = 16U;

  struct ProcessResult {
    std::size_t inputConsumed{0U};
    std::size_t outputProduced{0U};
    bool ok{false};
  };

  // Rates must be finite whole-Hz values. Exactly one side must be 44.1 kHz,
  // except for the exact 44.1 kHz bypass. The other side may be any whole-Hz
  // rate from 44.1 through 384 kHz.
  static bool isRatePairSupported(double inputRate, double outputRate) noexcept;

  // An optional sub-sample content delay, expressed on the output clock, can
  // align a downsample/model/upsample pair to an integer host sample. It shifts
  // the sinc center without changing output count or grid phase and must be in
  // [0, 1). The exact bypass accepts zero only.
  bool prepare(double inputRate,
               double outputRate,
               std::size_t channelCount,
               double additionalOutputDelaySamples = 0.0);

  // Reset to local sample zero on both clocks.
  void reset() noexcept;

  // Reset at an absolute input sample. The first output index is derived as
  // ceil(inputSampleIndex * outputRate / inputRate), preserving the absolute
  // rational clock phase without requiring a fractional timestamp.
  bool resetAtInputSample(std::uint64_t inputSampleIndex) noexcept;

  // Reset with an explicit next output index. Its source position must not
  // precede inputSampleIndex. A later output index intentionally creates a
  // gap; it will be emitted once its exact source position has arrived.
  bool resetAt(std::uint64_t inputSampleIndex,
               std::uint64_t outputSampleIndex) noexcept;

  // All input is consumed or none is. Call maxOutputForInput() first, or pass
  // at least that much capacity. Input and output buffers must contain the
  // configured number of non-null channels and must not overlap on a
  // converting path. Non-finite input is rejected atomically.
  ProcessResult process(const float* const* input,
                        std::size_t inputSamples,
                        float* const* output,
                        std::size_t outputCapacity) noexcept;

  // Exact number of output samples process() will produce for the next input
  // block in the current phase. Returns size_t::max on arithmetic overflow.
  std::size_t maxOutputForInput(std::size_t inputSamples) const noexcept;

  // Convert an absolute input index to the first output index whose undelayed
  // source position is not before it. Returns false on arithmetic overflow.
  bool mapInputToOutputCeil(std::uint64_t inputSampleIndex,
                            std::uint64_t& outputSampleIndex) const noexcept;

  bool isPrepared() const noexcept { return prepared_; }
  bool isBypassed() const noexcept { return bypassed_; }
  std::uint32_t inputRate() const noexcept { return inputRate_; }
  std::uint32_t outputRate() const noexcept { return outputRate_; }
  std::size_t channelCount() const noexcept { return channelCount_; }
  std::size_t filterLength() const noexcept { return filterLength_; }
  double cutoffFrequencyHz() const noexcept { return cutoffFrequencyHz_; }
  double additionalOutputDelaySamples() const noexcept {
    return additionalOutputDelaySamples_;
  }

  double filterGroupDelayInputSamples() const noexcept;
  double filterGroupDelayOutputSamples() const noexcept;
  double filterGroupDelaySeconds() const noexcept;

  std::uint64_t totalInputSamplesConsumed() const noexcept {
    return totalInputSamplesConsumed_;
  }
  std::uint64_t totalOutputSamplesProduced() const noexcept {
    return totalOutputSamplesProduced_;
  }
  std::uint64_t nextInputSampleIndex() const noexcept {
    return nextInputSampleIndex_;
  }
  std::uint64_t nextOutputSampleIndex() const noexcept {
    return nextOutputSampleIndex_;
  }

  // Exact source coordinate for nextOutputSampleIndex(): integer part plus
  // phaseNumerator / outputRate(). Exposing it makes block-to-block timestamp
  // and phase checks possible without floating-point accumulation.
  std::uint64_t nextOutputSourceSampleIndex() const noexcept {
    return nextOutputSourceSampleIndex_;
  }
  std::uint32_t nextOutputSourcePhaseNumerator() const noexcept {
    return nextOutputSourcePhaseNumerator_;
  }

private:
  static constexpr std::size_t kMaximumPolyphaseCount = 2048U;
  static constexpr std::size_t kMaximumFilterLength = 2048U;

  bool buildFilter();
  bool setOutputClock(std::uint64_t outputSampleIndex) noexcept;
  void advanceOutputClock() noexcept;
  float renderChannel(std::size_t channel) const noexcept;
  void clearState() noexcept;

  std::vector<float> history_;
  std::vector<float> coefficients_;

  std::uint32_t inputRate_{0U};
  std::uint32_t outputRate_{0U};
  std::uint32_t phaseDenominatorGcd_{1U};
  std::size_t channelCount_{0U};
  std::size_t filterLength_{0U};
  std::size_t phaseCount_{0U};
  std::size_t historyWriteIndex_{0U};
  double cutoffFrequencyHz_{0.0};
  double filterCenterInputSamples_{0.0};
  double additionalOutputDelaySamples_{0.0};

  std::uint64_t totalInputSamplesConsumed_{0U};
  std::uint64_t totalOutputSamplesProduced_{0U};
  std::uint64_t nextInputSampleIndex_{0U};
  std::uint64_t nextOutputSampleIndex_{0U};
  std::uint64_t nextOutputSourceSampleIndex_{0U};
  std::uint32_t nextOutputSourcePhaseNumerator_{0U};

  bool exactPhaseBank_{false};
  bool prepared_{false};
  bool bypassed_{false};
};

}  // namespace audio_plugin
