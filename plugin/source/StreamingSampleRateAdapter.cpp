#include "StemgenRT/StreamingSampleRateAdapter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <numeric>

namespace audio_plugin {
namespace {

constexpr double kPassbandEdgeHz = 20000.0;
constexpr double kStopbandEdgeHz = 0.5 * static_cast<double>(kModelSampleRate);
constexpr double kTargetStopbandAttenuationDb = 110.0;

bool rateToInteger(double rate, std::uint32_t& integerRate) noexcept {
  if (!std::isfinite(rate) || rate < 0.0 ||
      rate >
          static_cast<double>(StreamingSampleRateAdapter::kMaximumSampleRate)) {
    return false;
  }

  const double rounded = std::round(rate);
  if (std::abs(rate - rounded) > 1.0e-6) {
    return false;
  }
  integerRate = static_cast<std::uint32_t>(rounded);
  return true;
}

double modifiedBesselI0(double value) noexcept {
  const double quarterSquared = 0.25 * value * value;
  double sum = 1.0;
  double term = 1.0;
  for (std::uint32_t order = 1U; order <= 32U; ++order) {
    const double divisor =
        static_cast<double>(order) * static_cast<double>(order);
    term *= quarterSquared / divisor;
    sum += term;
    if (term <= sum * 1.0e-16) {
      break;
    }
  }
  return sum;
}

double normalizedSinc(double value) noexcept {
  if (std::abs(value) < 1.0e-12) {
    return 1.0;
  }
  const double angle = std::numbers::pi_v<double> * value;
  return std::sin(angle) / angle;
}

bool checkedCeilMulDiv(std::uint64_t value,
                       std::uint32_t multiplier,
                       std::uint32_t divisor,
                       std::uint64_t& result) noexcept {
  if (divisor == 0U) {
    return false;
  }

  const std::uint64_t divisor64 = static_cast<std::uint64_t>(divisor);
  const std::uint64_t multiplier64 = static_cast<std::uint64_t>(multiplier);
  const std::uint64_t quotient = value / divisor64;
  const std::uint64_t remainder = value % divisor64;
  if (quotient > std::numeric_limits<std::uint64_t>::max() / multiplier64) {
    return false;
  }

  const std::uint64_t whole = quotient * multiplier64;
  const std::uint64_t remainderProduct = remainder * multiplier64;
  const std::uint64_t fractional =
      remainderProduct / divisor64 +
      static_cast<std::uint64_t>(remainderProduct % divisor64 != 0U);
  if (whole > std::numeric_limits<std::uint64_t>::max() - fractional) {
    return false;
  }
  result = whole + fractional;
  return true;
}

}  // namespace

bool StreamingSampleRateAdapter::isRatePairSupported(
    double inputRate,
    double outputRate) noexcept {
  std::uint32_t integerInputRate = 0U;
  std::uint32_t integerOutputRate = 0U;
  if (!rateToInteger(inputRate, integerInputRate) ||
      !rateToInteger(outputRate, integerOutputRate)) {
    return false;
  }

  constexpr std::uint32_t modelSampleRate =
      static_cast<std::uint32_t>(kModelSampleRate);
  if (integerInputRate < modelSampleRate ||
      integerOutputRate < modelSampleRate) {
    return false;
  }
  return integerInputRate == modelSampleRate ||
         integerOutputRate == modelSampleRate;
}

bool StreamingSampleRateAdapter::prepare(double inputRate,
                                         double outputRate,
                                         std::size_t channelCount,
                                         double additionalOutputDelaySamples) {
  prepared_ = false;
  if (!isRatePairSupported(inputRate, outputRate) || channelCount == 0U ||
      channelCount > kMaximumChannelCount ||
      !std::isfinite(additionalOutputDelaySamples) ||
      additionalOutputDelaySamples < 0.0 ||
      additionalOutputDelaySamples >= 1.0) {
    return false;
  }

  static_cast<void>(rateToInteger(inputRate, inputRate_));
  static_cast<void>(rateToInteger(outputRate, outputRate_));
  channelCount_ = channelCount;
  bypassed_ = inputRate_ == outputRate_;
  if (bypassed_ && additionalOutputDelaySamples > 0.0) {
    return false;
  }
  additionalOutputDelaySamples_ = additionalOutputDelaySamples;
  cutoffFrequencyHz_ =
      bypassed_ ? 0.0 : 0.5 * (kPassbandEdgeHz + kStopbandEdgeHz);

  if (bypassed_) {
    filterLength_ = 0U;
    phaseCount_ = 0U;
    phaseDenominatorGcd_ = inputRate_;
    exactPhaseBank_ = true;
    filterCenterInputSamples_ = 0.0;
    history_.clear();
    coefficients_.clear();
  } else if (!buildFilter()) {
    return false;
  }

  prepared_ = true;
  reset();
  return true;
}

bool StreamingSampleRateAdapter::buildFilter() {
  const double transitionWidthHz = kStopbandEdgeHz - kPassbandEdgeHz;
  const double normalizedTransition =
      transitionWidthHz / static_cast<double>(inputRate_);
  const double estimatedOrder =
      (kTargetStopbandAttenuationDb - 8.0) /
      (2.285 * 2.0 * std::numbers::pi_v<double> * normalizedTransition);
  std::size_t requestedLength =
      static_cast<std::size_t>(std::ceil(estimatedOrder + 1.0));
  requestedLength = std::max<std::size_t>(128U, requestedLength);
  if ((requestedLength % 2U) != 0U) {
    ++requestedLength;
  }
  if (requestedLength > kMaximumFilterLength) {
    return false;
  }
  filterLength_ = requestedLength;

  phaseDenominatorGcd_ = std::gcd(inputRate_, outputRate_);
  const std::size_t exactPhaseCount =
      static_cast<std::size_t>(outputRate_ / phaseDenominatorGcd_);
  phaseCount_ = std::min(exactPhaseCount, kMaximumPolyphaseCount);
  exactPhaseBank_ = exactPhaseCount <= kMaximumPolyphaseCount;

  const std::size_t historySamplesPerChannel = 2U * filterLength_;
  if (channelCount_ >
          std::numeric_limits<std::size_t>::max() / historySamplesPerChannel ||
      phaseCount_ + 1U >
          std::numeric_limits<std::size_t>::max() / filterLength_) {
    return false;
  }

  try {
    history_.assign(channelCount_ * historySamplesPerChannel, 0.0f);
    coefficients_.resize((phaseCount_ + 1U) * filterLength_);
  } catch (...) {
    history_.clear();
    coefficients_.clear();
    return false;
  }

  constexpr double beta = 0.1102 * (kTargetStopbandAttenuationDb - 8.7);
  const double inverseBesselBeta = 1.0 / modifiedBesselI0(beta);
  const double halfLength = 0.5 * static_cast<double>(filterLength_ - 1U);
  filterCenterInputSamples_ = halfLength + additionalOutputDelaySamples_ *
                                               static_cast<double>(inputRate_) /
                                               static_cast<double>(outputRate_);
  const double normalizedCutoff =
      cutoffFrequencyHz_ / static_cast<double>(inputRate_);

  for (std::size_t phase = 0U; phase <= phaseCount_; ++phase) {
    const double fraction =
        static_cast<double>(phase) / static_cast<double>(phaseCount_);
    double coefficientSum = 0.0;
    for (std::size_t tap = 0U; tap < filterLength_; ++tap) {
      const double distance =
          static_cast<double>(tap) + fraction - filterCenterInputSamples_;
      const double normalizedDistance = distance / halfLength;
      double window = 0.0;
      if (std::abs(normalizedDistance) <= 1.0) {
        const double windowArgument =
            beta * std::sqrt(std::max(
                       0.0, 1.0 - normalizedDistance * normalizedDistance));
        window = modifiedBesselI0(windowArgument) * inverseBesselBeta;
      }
      coefficientSum += 2.0 * normalizedCutoff *
                        normalizedSinc(2.0 * normalizedCutoff * distance) *
                        window;
    }
    if (!std::isfinite(coefficientSum) || std::abs(coefficientSum) < 1.0e-12) {
      return false;
    }

    const std::size_t phaseOffset = phase * filterLength_;
    for (std::size_t tap = 0U; tap < filterLength_; ++tap) {
      const double distance =
          static_cast<double>(tap) + fraction - filterCenterInputSamples_;
      const double normalizedDistance = distance / halfLength;
      double window = 0.0;
      if (std::abs(normalizedDistance) <= 1.0) {
        const double windowArgument =
            beta * std::sqrt(std::max(
                       0.0, 1.0 - normalizedDistance * normalizedDistance));
        window = modifiedBesselI0(windowArgument) * inverseBesselBeta;
      }
      const double coefficient =
          (2.0 * normalizedCutoff *
           normalizedSinc(2.0 * normalizedCutoff * distance) * window) /
          coefficientSum;
      // The mirrored history is chronological, oldest to newest. Reverse the
      // tap bank once here so both operands advance forward in the hot loop.
      const std::size_t reversedTap = filterLength_ - 1U - tap;
      coefficients_[phaseOffset + reversedTap] =
          static_cast<float>(coefficient);
    }
  }
  return true;
}

void StreamingSampleRateAdapter::reset() noexcept {
  static_cast<void>(resetAt(0U, 0U));
}

bool StreamingSampleRateAdapter::resetAtInputSample(
    std::uint64_t inputSampleIndex) noexcept {
  std::uint64_t outputSampleIndex = 0U;
  if (!mapInputToOutputCeil(inputSampleIndex, outputSampleIndex)) {
    return false;
  }
  return resetAt(inputSampleIndex, outputSampleIndex);
}

bool StreamingSampleRateAdapter::resetAt(
    std::uint64_t inputSampleIndex,
    std::uint64_t outputSampleIndex) noexcept {
  if (!prepared_) {
    return false;
  }

  const std::uint64_t previousOutputSampleIndex = nextOutputSampleIndex_;
  const std::uint64_t previousSourceSampleIndex = nextOutputSourceSampleIndex_;
  const std::uint32_t previousPhaseNumerator = nextOutputSourcePhaseNumerator_;
  if (!setOutputClock(outputSampleIndex) ||
      nextOutputSourceSampleIndex_ < inputSampleIndex) {
    nextOutputSampleIndex_ = previousOutputSampleIndex;
    nextOutputSourceSampleIndex_ = previousSourceSampleIndex;
    nextOutputSourcePhaseNumerator_ = previousPhaseNumerator;
    return false;
  }

  clearState();
  nextInputSampleIndex_ = inputSampleIndex;
  nextOutputSampleIndex_ = outputSampleIndex;
  totalInputSamplesConsumed_ = 0U;
  totalOutputSamplesProduced_ = 0U;
  return true;
}

void StreamingSampleRateAdapter::clearState() noexcept {
  std::fill(history_.begin(), history_.end(), 0.0f);
  historyWriteIndex_ = 0U;
}

bool StreamingSampleRateAdapter::setOutputClock(
    std::uint64_t outputSampleIndex) noexcept {
  if (outputRate_ == 0U) {
    return false;
  }

  const std::uint64_t outputRate64 = static_cast<std::uint64_t>(outputRate_);
  const std::uint64_t inputRate64 = static_cast<std::uint64_t>(inputRate_);
  const std::uint64_t quotient = outputSampleIndex / outputRate64;
  const std::uint64_t remainder = outputSampleIndex % outputRate64;
  if (quotient > std::numeric_limits<std::uint64_t>::max() / inputRate64) {
    return false;
  }
  const std::uint64_t whole = quotient * inputRate64;
  const std::uint64_t remainderProduct = remainder * inputRate64;
  const std::uint64_t fractionalWhole = remainderProduct / outputRate64;
  if (whole > std::numeric_limits<std::uint64_t>::max() - fractionalWhole) {
    return false;
  }

  nextOutputSampleIndex_ = outputSampleIndex;
  nextOutputSourceSampleIndex_ = whole + fractionalWhole;
  nextOutputSourcePhaseNumerator_ =
      static_cast<std::uint32_t>(remainderProduct % outputRate64);
  return true;
}

bool StreamingSampleRateAdapter::mapInputToOutputCeil(
    std::uint64_t inputSampleIndex,
    std::uint64_t& outputSampleIndex) const noexcept {
  if (!prepared_) {
    return false;
  }
  return checkedCeilMulDiv(inputSampleIndex, outputRate_, inputRate_,
                           outputSampleIndex);
}

std::size_t StreamingSampleRateAdapter::maxOutputForInput(
    std::size_t inputSamples) const noexcept {
  if (!prepared_ || inputSamples == 0U) {
    return 0U;
  }
  if (static_cast<std::uint64_t>(inputSamples) >
      std::numeric_limits<std::uint64_t>::max() - nextInputSampleIndex_) {
    return std::numeric_limits<std::size_t>::max();
  }

  const std::uint64_t inputEndExclusive =
      nextInputSampleIndex_ + static_cast<std::uint64_t>(inputSamples);
  std::uint64_t outputEndExclusive = 0U;
  if (!checkedCeilMulDiv(inputEndExclusive, outputRate_, inputRate_,
                         outputEndExclusive)) {
    return std::numeric_limits<std::size_t>::max();
  }
  if (outputEndExclusive <= nextOutputSampleIndex_) {
    return 0U;
  }
  const std::uint64_t outputCount = outputEndExclusive - nextOutputSampleIndex_;
  if (outputCount >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return std::numeric_limits<std::size_t>::max();
  }
  return static_cast<std::size_t>(outputCount);
}

StreamingSampleRateAdapter::ProcessResult StreamingSampleRateAdapter::process(
    const float* const* input,
    std::size_t inputSamples,
    float* const* output,
    std::size_t outputCapacity) noexcept {
  ProcessResult result;
  if (!prepared_) {
    return result;
  }
  if (inputSamples == 0U) {
    result.ok = true;
    return result;
  }

  const std::size_t requiredOutput = maxOutputForInput(inputSamples);
  if (requiredOutput == std::numeric_limits<std::size_t>::max() ||
      outputCapacity < requiredOutput || input == nullptr ||
      output == nullptr ||
      static_cast<std::uint64_t>(inputSamples) >
          std::numeric_limits<std::uint64_t>::max() -
              totalInputSamplesConsumed_ ||
      static_cast<std::uint64_t>(requiredOutput) >
          std::numeric_limits<std::uint64_t>::max() -
              totalOutputSamplesProduced_) {
    return result;
  }

  for (std::size_t channel = 0U; channel < channelCount_; ++channel) {
    if (input[channel] == nullptr ||
        (requiredOutput > 0U && output[channel] == nullptr)) {
      return result;
    }
    for (std::size_t sample = 0U; sample < inputSamples; ++sample) {
      if (!std::isfinite(input[channel][sample])) {
        return result;
      }
    }
  }

  if (bypassed_) {
    const std::uint64_t skippedInput64 =
        nextOutputSampleIndex_ - nextInputSampleIndex_;
    const std::size_t skippedInput = static_cast<std::size_t>(
        std::min(skippedInput64, static_cast<std::uint64_t>(inputSamples)));
    const std::size_t byteCount = requiredOutput * sizeof(float);
    for (std::size_t channel = 0U; channel < channelCount_; ++channel) {
      if (requiredOutput > 0U) {
        std::memmove(output[channel], input[channel] + skippedInput, byteCount);
      }
    }
    result.inputConsumed = inputSamples;
    result.outputProduced = requiredOutput;
    result.ok = true;
    totalInputSamplesConsumed_ += static_cast<std::uint64_t>(inputSamples);
    totalOutputSamplesProduced_ += static_cast<std::uint64_t>(requiredOutput);
    nextInputSampleIndex_ += static_cast<std::uint64_t>(inputSamples);
    nextOutputSampleIndex_ += static_cast<std::uint64_t>(requiredOutput);
    nextOutputSourceSampleIndex_ = nextOutputSampleIndex_;
    nextOutputSourcePhaseNumerator_ = 0U;
    return result;
  }

  std::size_t produced = 0U;
  const std::size_t historyStride = 2U * filterLength_;
  for (std::size_t sample = 0U; sample < inputSamples; ++sample) {
    for (std::size_t channel = 0U; channel < channelCount_; ++channel) {
      const std::size_t channelOffset = channel * historyStride;
      const float value = input[channel][sample];
      history_[channelOffset + historyWriteIndex_] = value;
      history_[channelOffset + historyWriteIndex_ + filterLength_] = value;
    }

    ++historyWriteIndex_;
    if (historyWriteIndex_ == filterLength_) {
      historyWriteIndex_ = 0U;
    }

    const std::uint64_t currentInputSampleIndex = nextInputSampleIndex_;
    ++nextInputSampleIndex_;
    while (nextOutputSourceSampleIndex_ <= currentInputSampleIndex) {
      for (std::size_t channel = 0U; channel < channelCount_; ++channel) {
        output[channel][produced] = renderChannel(channel);
      }
      ++produced;
      ++totalOutputSamplesProduced_;
      advanceOutputClock();
    }
  }

  totalInputSamplesConsumed_ += static_cast<std::uint64_t>(inputSamples);
  result.inputConsumed = inputSamples;
  result.outputProduced = produced;
  result.ok = produced == requiredOutput;
  return result;
}

float StreamingSampleRateAdapter::renderChannel(
    std::size_t channel) const noexcept {
  const std::size_t historyOffset = channel * 2U * filterLength_;
  const float* const history =
      history_.data() + historyOffset + historyWriteIndex_;

  std::size_t phaseIndex = 0U;
  float interpolation = 0.0f;
  if (exactPhaseBank_) {
    phaseIndex = static_cast<std::size_t>(nextOutputSourcePhaseNumerator_ /
                                          phaseDenominatorGcd_);
  } else {
    const std::uint64_t scaledPhase =
        static_cast<std::uint64_t>(nextOutputSourcePhaseNumerator_) *
        static_cast<std::uint64_t>(phaseCount_);
    phaseIndex = static_cast<std::size_t>(
        scaledPhase / static_cast<std::uint64_t>(outputRate_));
    interpolation = static_cast<float>(
        static_cast<double>(scaledPhase %
                            static_cast<std::uint64_t>(outputRate_)) /
        static_cast<double>(outputRate_));
  }

  const float* const firstCoefficients =
      coefficients_.data() + phaseIndex * filterLength_;
  float sum = 0.0f;
  if (exactPhaseBank_) {
    for (std::size_t tap = 0U; tap < filterLength_; ++tap) {
      sum += history[tap] * firstCoefficients[tap];
    }
  } else {
    const float* const secondCoefficients = firstCoefficients + filterLength_;
    for (std::size_t tap = 0U; tap < filterLength_; ++tap) {
      const float coefficient =
          firstCoefficients[tap] +
          interpolation * (secondCoefficients[tap] - firstCoefficients[tap]);
      sum += history[tap] * coefficient;
    }
  }
  return sum;
}

void StreamingSampleRateAdapter::advanceOutputClock() noexcept {
  const std::uint64_t phase =
      static_cast<std::uint64_t>(nextOutputSourcePhaseNumerator_) +
      static_cast<std::uint64_t>(inputRate_);
  nextOutputSourceSampleIndex_ +=
      phase / static_cast<std::uint64_t>(outputRate_);
  nextOutputSourcePhaseNumerator_ = static_cast<std::uint32_t>(
      phase % static_cast<std::uint64_t>(outputRate_));
  ++nextOutputSampleIndex_;
}

double StreamingSampleRateAdapter::filterGroupDelayInputSamples()
    const noexcept {
  if (bypassed_ || filterLength_ == 0U) {
    return 0.0;
  }
  return filterCenterInputSamples_;
}

double StreamingSampleRateAdapter::filterGroupDelayOutputSamples()
    const noexcept {
  if (!prepared_ || inputRate_ == 0U) {
    return 0.0;
  }
  return filterGroupDelayInputSamples() * static_cast<double>(outputRate_) /
         static_cast<double>(inputRate_);
}

double StreamingSampleRateAdapter::filterGroupDelaySeconds() const noexcept {
  if (!prepared_ || inputRate_ == 0U) {
    return 0.0;
  }
  return filterGroupDelayInputSamples() / static_cast<double>(inputRate_);
}

}  // namespace audio_plugin
