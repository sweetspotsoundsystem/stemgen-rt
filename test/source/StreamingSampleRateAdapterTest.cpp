#include <StemgenRT/StreamingSampleRateAdapter.h>
#include <gtest/gtest.h>
#include <bit>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

namespace {

using audio_plugin::StreamingSampleRateAdapter;
constexpr std::uint32_t kTestModelSampleRate =
    static_cast<std::uint32_t>(audio_plugin::kModelSampleRate);
constexpr double kTestModelSampleRateDouble =
    static_cast<double>(audio_plugin::kModelSampleRate);

struct ConvertedAudio {
  std::vector<std::vector<float>> channels;
  std::uint64_t outputStart{0U};
};

std::uint64_t ceilMulDiv(std::uint64_t value,
                         std::uint32_t multiplier,
                         std::uint32_t divisor) {
  const std::uint64_t product = value * static_cast<std::uint64_t>(multiplier);
  const std::uint64_t divisor64 = static_cast<std::uint64_t>(divisor);
  return product / divisor64 +
         static_cast<std::uint64_t>(product % divisor64 != 0U);
}

ConvertedAudio convertInBlocks(StreamingSampleRateAdapter& adapter,
                               const std::vector<std::vector<float>>& input,
                               const std::vector<std::size_t>& blockPattern) {
  ConvertedAudio converted;
  converted.channels.resize(input.size());
  converted.outputStart = adapter.nextOutputSampleIndex();

  std::size_t inputOffset = 0U;
  std::size_t patternIndex = 0U;
  while (inputOffset < input.front().size()) {
    const std::size_t requested =
        blockPattern[patternIndex % blockPattern.size()];
    const std::size_t blockSize =
        std::min(requested, input.front().size() - inputOffset);
    const std::size_t outputSize = adapter.maxOutputForInput(blockSize);
    std::vector<std::vector<float>> output(
        input.size(), std::vector<float>(outputSize, 0.0f));
    std::vector<const float*> inputPointers(input.size(), nullptr);
    std::vector<float*> outputPointers(input.size(), nullptr);
    for (std::size_t channel = 0U; channel < input.size(); ++channel) {
      inputPointers[channel] = input[channel].data() + inputOffset;
      outputPointers[channel] = output[channel].data();
    }

    const auto result = adapter.process(inputPointers.data(), blockSize,
                                        outputPointers.data(), outputSize);
    EXPECT_TRUE(result.ok);
    EXPECT_EQ(result.inputConsumed, blockSize);
    EXPECT_EQ(result.outputProduced, outputSize);
    for (std::size_t channel = 0U; channel < input.size(); ++channel) {
      converted.channels[channel].insert(converted.channels[channel].end(),
                                         output[channel].begin(),
                                         output[channel].end());
    }
    inputOffset += blockSize;
    ++patternIndex;
  }
  return converted;
}

std::vector<std::vector<float>> makeSignal(std::size_t channels,
                                           std::size_t samples,
                                           double sampleRate) {
  std::vector<std::vector<float>> signal(channels,
                                         std::vector<float>(samples, 0.0f));
  for (std::size_t sample = 0U; sample < samples; ++sample) {
    const double time = static_cast<double>(sample) / sampleRate;
    signal[0][sample] = static_cast<float>(
        0.55 * std::sin(2.0 * std::numbers::pi_v<double> * 997.0 * time) +
        0.2 * std::sin(2.0 * std::numbers::pi_v<double> * 13001.0 * time));
    if (channels > 1U) {
      signal[1][sample] = static_cast<float>(
          0.4 * std::cos(2.0 * std::numbers::pi_v<double> * 1801.0 * time) -
          0.25 * std::sin(2.0 * std::numbers::pi_v<double> * 17003.0 * time));
    }
  }
  return signal;
}

std::vector<std::vector<float>> makeSine(double frequency,
                                         double sampleRate,
                                         std::size_t samples) {
  std::vector<std::vector<float>> signal(1U, std::vector<float>(samples, 0.0f));
  for (std::size_t sample = 0U; sample < samples; ++sample) {
    signal[0][sample] = static_cast<float>(
        std::sin(2.0 * std::numbers::pi_v<double> * frequency *
                 static_cast<double>(sample) / sampleRate));
  }
  return signal;
}

double measuredAmplitude(const std::vector<float>& signal,
                         double frequency,
                         double sampleRate,
                         std::size_t discardAtEachEnd) {
  EXPECT_GT(signal.size(), 2U * discardAtEachEnd);
  const std::size_t begin = discardAtEachEnd;
  const std::size_t end = signal.size() - discardAtEachEnd;
  double cosineProjection = 0.0;
  double sineProjection = 0.0;
  for (std::size_t sample = begin; sample < end; ++sample) {
    const double phase = 2.0 * std::numbers::pi_v<double> * frequency *
                         static_cast<double>(sample) / sampleRate;
    cosineProjection += static_cast<double>(signal[sample]) * std::cos(phase);
    sineProjection += static_cast<double>(signal[sample]) * std::sin(phase);
  }
  const double count = static_cast<double>(end - begin);
  return 2.0 * std::hypot(cosineProjection, sineProjection) / count;
}

}  // namespace

TEST(StreamingSampleRateAdapterTest,
     SupportsModelClockPairsAndBypassesExactlyAt44100) {
  EXPECT_TRUE(StreamingSampleRateAdapter::isRatePairSupported(
      kTestModelSampleRateDouble, kTestModelSampleRateDouble));
  EXPECT_TRUE(StreamingSampleRateAdapter::isRatePairSupported(
      48000.0, kTestModelSampleRateDouble));
  EXPECT_TRUE(StreamingSampleRateAdapter::isRatePairSupported(
      kTestModelSampleRateDouble, 192000.0));
  EXPECT_FALSE(
      StreamingSampleRateAdapter::isRatePairSupported(48000.0, 96000.0));
  EXPECT_FALSE(StreamingSampleRateAdapter::isRatePairSupported(
      kTestModelSampleRateDouble - 1.0, kTestModelSampleRateDouble));
  EXPECT_FALSE(StreamingSampleRateAdapter::isRatePairSupported(
      kTestModelSampleRateDouble + 0.5, kTestModelSampleRateDouble));

  StreamingSampleRateAdapter adapter;
  ASSERT_TRUE(adapter.prepare(kTestModelSampleRateDouble,
                              kTestModelSampleRateDouble, 2U));
  EXPECT_TRUE(adapter.isBypassed());
  EXPECT_EQ(adapter.filterLength(), 0U);
  EXPECT_DOUBLE_EQ(adapter.filterGroupDelaySeconds(), 0.0);

  std::array<std::vector<float>, 2U> input = {
      std::vector<float>{0.0f, -0.0f, 0.125f, -0.75f, 1.0f},
      std::vector<float>{-1.0f, 0.5f, -0.25f, 0.0f, -0.0f}};
  std::array<std::vector<float>, 2U> output = {
      std::vector<float>(input[0].size(), 0.0f),
      std::vector<float>(input[1].size(), 0.0f)};
  const std::array<const float*, 2U> inputPointers = {input[0].data(),
                                                      input[1].data()};
  const std::array<float*, 2U> outputPointers = {output[0].data(),
                                                 output[1].data()};
  const auto result = adapter.process(inputPointers.data(), input[0].size(),
                                      outputPointers.data(), output[0].size());
  ASSERT_TRUE(result.ok);
  EXPECT_EQ(result.outputProduced, input[0].size());
  for (std::size_t channel = 0U; channel < input.size(); ++channel) {
    for (std::size_t sample = 0U; sample < input[channel].size(); ++sample) {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(output[channel][sample]),
                std::bit_cast<std::uint32_t>(input[channel][sample]));
    }
  }
}

TEST(StreamingSampleRateAdapterTest,
     CountAndRationalPhaseDoNotDriftAcrossIrregularBlocks) {
  constexpr std::uint32_t inputRate = 48000U;
  constexpr std::uint32_t outputRate = kTestModelSampleRate;
  constexpr std::size_t inputSampleCount = 480123U;
  StreamingSampleRateAdapter adapter;
  ASSERT_TRUE(adapter.prepare(static_cast<double>(inputRate),
                              static_cast<double>(outputRate), 1U));

  const std::vector<std::vector<float>> input(
      1U, std::vector<float>(inputSampleCount, 0.0f));
  const auto converted =
      convertInBlocks(adapter, input, {1U, 7U, 64U, 511U, 3U, 1024U, 127U});
  const std::uint64_t expectedOutputCount =
      ceilMulDiv(inputSampleCount, outputRate, inputRate);
  EXPECT_EQ(converted.channels[0].size(), expectedOutputCount);
  EXPECT_EQ(adapter.totalInputSamplesConsumed(), inputSampleCount);
  EXPECT_EQ(adapter.totalOutputSamplesProduced(), expectedOutputCount);
  EXPECT_EQ(adapter.nextInputSampleIndex(), inputSampleCount);
  EXPECT_EQ(adapter.nextOutputSampleIndex(), expectedOutputCount);

  const std::uint64_t phaseProduct =
      expectedOutputCount * static_cast<std::uint64_t>(inputRate);
  EXPECT_EQ(adapter.nextOutputSourceSampleIndex(), phaseProduct / outputRate);
  EXPECT_EQ(adapter.nextOutputSourcePhaseNumerator(),
            phaseProduct % outputRate);
}

TEST(StreamingSampleRateAdapterTest,
     OutputIsIndependentOfInputBlockBoundaries) {
  const auto input = makeSignal(2U, 12000U, 96000.0);
  StreamingSampleRateAdapter contiguousAdapter;
  StreamingSampleRateAdapter fragmentedAdapter;
  ASSERT_TRUE(
      contiguousAdapter.prepare(96000.0, kTestModelSampleRateDouble, 2U));
  ASSERT_TRUE(
      fragmentedAdapter.prepare(96000.0, kTestModelSampleRateDouble, 2U));

  const auto contiguous =
      convertInBlocks(contiguousAdapter, input, {input[0].size()});
  const auto fragmented = convertInBlocks(fragmentedAdapter, input,
                                          {1U, 2U, 17U, 255U, 3U, 512U, 79U});
  ASSERT_EQ(contiguous.channels, fragmented.channels);
}

TEST(StreamingSampleRateAdapterTest,
     ResetIsDeterministicAndCanResumeAtAbsoluteRationalPhase) {
  const auto input = makeSignal(1U, 4096U, 48000.0);
  StreamingSampleRateAdapter adapter;
  ASSERT_TRUE(adapter.prepare(48000.0, kTestModelSampleRateDouble, 1U));
  const auto first = convertInBlocks(adapter, input, {37U, 511U, 9U});
  adapter.reset();
  const auto second = convertInBlocks(adapter, input, {1024U, 3U});
  EXPECT_EQ(first.channels, second.channels);

  constexpr std::uint64_t absoluteInputOrigin = 1234567U;
  ASSERT_TRUE(adapter.resetAtInputSample(absoluteInputOrigin));
  const std::uint64_t expectedOutputOrigin =
      ceilMulDiv(absoluteInputOrigin, kTestModelSampleRate, 48000U);
  EXPECT_EQ(adapter.nextInputSampleIndex(), absoluteInputOrigin);
  EXPECT_EQ(adapter.nextOutputSampleIndex(), expectedOutputOrigin);
  EXPECT_GE(adapter.nextOutputSourceSampleIndex(), absoluteInputOrigin);
  EXPECT_LT(adapter.nextOutputSourceSampleIndex(), absoluteInputOrigin + 2U);
  EXPECT_EQ(adapter.totalInputSamplesConsumed(), 0U);
  EXPECT_EQ(adapter.totalOutputSamplesProduced(), 0U);

  std::uint64_t mappedOutput = 0U;
  ASSERT_TRUE(adapter.mapInputToOutputCeil(absoluteInputOrigin, mappedOutput));
  EXPECT_EQ(mappedOutput, expectedOutputOrigin);
  const std::uint64_t inputIndexBeforeInvalidReset =
      adapter.nextInputSampleIndex();
  const std::uint64_t outputIndexBeforeInvalidReset =
      adapter.nextOutputSampleIndex();
  EXPECT_FALSE(adapter.resetAt(absoluteInputOrigin, expectedOutputOrigin - 1U));
  EXPECT_EQ(adapter.nextInputSampleIndex(), inputIndexBeforeInvalidReset);
  EXPECT_EQ(adapter.nextOutputSampleIndex(), outputIndexBeforeInvalidReset);
}

TEST(StreamingSampleRateAdapterTest,
     InvalidInputAndInsufficientCapacityLeaveStreamingStateUntouched) {
  StreamingSampleRateAdapter adapter;
  StreamingSampleRateAdapter control;
  ASSERT_TRUE(adapter.prepare(48000.0, kTestModelSampleRateDouble, 1U));
  ASSERT_TRUE(control.prepare(48000.0, kTestModelSampleRateDouble, 1U));

  const auto warmup = makeSignal(1U, 1000U, 48000.0);
  static_cast<void>(convertInBlocks(adapter, warmup, {1000U}));
  static_cast<void>(convertInBlocks(control, warmup, {1000U}));
  const std::uint64_t inputBefore = adapter.nextInputSampleIndex();
  const std::uint64_t outputBefore = adapter.nextOutputSampleIndex();

  const std::vector<float> nonFinite = {
      0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f};
  std::vector<float> rejectedOutput(8U, 123.0f);
  const float* invalidInputPointer = nonFinite.data();
  float* rejectedOutputPointer = rejectedOutput.data();
  const auto invalidResult =
      adapter.process(&invalidInputPointer, nonFinite.size(),
                      &rejectedOutputPointer, rejectedOutput.size());
  EXPECT_FALSE(invalidResult.ok);
  EXPECT_EQ(adapter.nextInputSampleIndex(), inputBefore);
  EXPECT_EQ(adapter.nextOutputSampleIndex(), outputBefore);
  EXPECT_TRUE(std::all_of(rejectedOutput.begin(), rejectedOutput.end(),
                          [](float value) {
                            return std::bit_cast<uint32_t>(value) ==
                                   std::bit_cast<uint32_t>(123.0f);
                          }));

  const auto continuation = makeSignal(1U, 500U, 48000.0);
  const std::size_t required =
      adapter.maxOutputForInput(continuation[0].size());
  std::vector<float> tooSmall(required - 1U, 0.0f);
  const float* continuationPointer = continuation[0].data();
  float* tooSmallPointer = tooSmall.data();
  const auto capacityResult =
      adapter.process(&continuationPointer, continuation[0].size(),
                      &tooSmallPointer, tooSmall.size());
  EXPECT_FALSE(capacityResult.ok);
  EXPECT_EQ(adapter.nextInputSampleIndex(), inputBefore);
  EXPECT_EQ(adapter.nextOutputSampleIndex(), outputBefore);

  const auto actual = convertInBlocks(adapter, continuation, {11U, 256U});
  const auto expected = convertInBlocks(control, continuation, {500U});
  EXPECT_EQ(actual.channels, expected.channels);
  EXPECT_TRUE(std::all_of(actual.channels[0].begin(), actual.channels[0].end(),
                          [](float value) { return std::isfinite(value); }));
}

TEST(StreamingSampleRateAdapterTest,
     ImpulsePeakMatchesReportedLinearPhaseGroupDelay) {
  StreamingSampleRateAdapter adapter;
  ASSERT_TRUE(adapter.prepare(96000.0, kTestModelSampleRateDouble, 1U));
  std::vector<std::vector<float>> impulse(
      1U, std::vector<float>(adapter.filterLength() + 512U, 0.0f));
  impulse[0][0] = 1.0f;
  const auto output = convertInBlocks(adapter, impulse, {19U, 257U, 3U});
  const auto maximum = std::max_element(
      output.channels[0].begin(), output.channels[0].end(),
      [](float left, float right) { return std::abs(left) < std::abs(right); });
  ASSERT_NE(maximum, output.channels[0].end());
  const std::size_t peakIndex = static_cast<std::size_t>(
      std::distance(output.channels[0].begin(), maximum));
  EXPECT_NEAR(static_cast<double>(peakIndex),
              adapter.filterGroupDelayOutputSamples(), 1.0);
  EXPECT_NEAR(
      adapter.filterGroupDelaySeconds(),
      adapter.filterGroupDelayOutputSamples() / kTestModelSampleRateDouble,
      1.0e-12);
}

TEST(StreamingSampleRateAdapterTest,
     FractionalOutputDelayCanMakePairedHostDelayExactlyIntegral) {
  constexpr double modelRate = kTestModelSampleRateDouble;
  for (const double hostRate :
       {48000.0, 88200.0, 96000.0, 176400.0, 192000.0}) {
    SCOPED_TRACE(::testing::Message() << "hostRate=" << hostRate);
    StreamingSampleRateAdapter inputAdapter;
    StreamingSampleRateAdapter baseOutputAdapter;
    ASSERT_TRUE(inputAdapter.prepare(hostRate, modelRate, 2U));
    ASSERT_TRUE(baseOutputAdapter.prepare(modelRate, hostRate, 6U));

    const double basePairedHostDelay =
        inputAdapter.filterGroupDelayInputSamples() +
        baseOutputAdapter.filterGroupDelayOutputSamples();
    const double integralHostDelay = std::ceil(basePairedHostDelay);
    const double fractionalCorrection = integralHostDelay - basePairedHostDelay;
    ASSERT_GE(fractionalCorrection, 0.0);
    ASSERT_LT(fractionalCorrection, 1.0);

    StreamingSampleRateAdapter adjustedOutputAdapter;
    ASSERT_TRUE(adjustedOutputAdapter.prepare(modelRate, hostRate, 6U,
                                              fractionalCorrection));
    EXPECT_DOUBLE_EQ(adjustedOutputAdapter.additionalOutputDelaySamples(),
                     fractionalCorrection);
    const double adjustedPairedHostDelay =
        inputAdapter.filterGroupDelayInputSamples() +
        adjustedOutputAdapter.filterGroupDelayOutputSamples();
    EXPECT_NEAR(adjustedPairedHostDelay, integralHostDelay, 1.0e-12);

    // Content-delay adjustment must not move the rational sampling grid.
    EXPECT_EQ(adjustedOutputAdapter.maxOutputForInput(512U),
              baseOutputAdapter.maxOutputForInput(512U));
    EXPECT_EQ(adjustedOutputAdapter.nextOutputSourceSampleIndex(),
              baseOutputAdapter.nextOutputSourceSampleIndex());
    EXPECT_EQ(adjustedOutputAdapter.nextOutputSourcePhaseNumerator(),
              baseOutputAdapter.nextOutputSourcePhaseNumerator());
  }
}

TEST(StreamingSampleRateAdapterTest,
     PairedImpulsePeakMatchesReportedIntegerHostDelayAtEveryQualifiedRate) {
  constexpr double modelRate = kTestModelSampleRateDouble;
  for (const double hostRate :
       {48000.0, 88200.0, 96000.0, 176400.0, 192000.0}) {
    SCOPED_TRACE(::testing::Message() << "hostRate=" << hostRate);
    StreamingSampleRateAdapter inputAdapter;
    StreamingSampleRateAdapter baseOutputAdapter;
    ASSERT_TRUE(inputAdapter.prepare(hostRate, modelRate, 1U));
    ASSERT_TRUE(baseOutputAdapter.prepare(modelRate, hostRate, 1U));

    const double naturalDelay =
        inputAdapter.filterGroupDelayInputSamples() +
        baseOutputAdapter.filterGroupDelayOutputSamples();
    const double expectedIntegerDelay = std::ceil(naturalDelay - 1.0e-9);
    const double fractionalCorrection = expectedIntegerDelay - naturalDelay;

    StreamingSampleRateAdapter outputAdapter;
    ASSERT_TRUE(outputAdapter.prepare(modelRate, hostRate, 1U,
                                      std::max(0.0, fractionalCorrection)));
    ASSERT_NEAR(inputAdapter.filterGroupDelayInputSamples() +
                    outputAdapter.filterGroupDelayOutputSamples(),
                expectedIntegerDelay, 1.0e-12);

    const std::size_t hostSampleCount =
        static_cast<std::size_t>(hostRate / 4.0);
    std::vector<std::vector<float>> hostImpulse(
        1U, std::vector<float>(hostSampleCount, 0.0f));
    hostImpulse[0][0] = 1.0f;
    const ConvertedAudio modelImpulse =
        convertInBlocks(inputAdapter, hostImpulse, {1U, 37U, 511U, 3U, 1024U});
    const ConvertedAudio hostOutput = convertInBlocks(
        outputAdapter, modelImpulse.channels, {7U, 512U, 29U, 1U});

    ASSERT_EQ(modelImpulse.outputStart, 0U);
    ASSERT_EQ(hostOutput.outputStart, 0U);
    const auto maximum = std::max_element(
        hostOutput.channels[0].begin(), hostOutput.channels[0].end(),
        [](float left, float right) {
          return std::abs(left) < std::abs(right);
        });
    ASSERT_NE(maximum, hostOutput.channels[0].end());
    const std::size_t peakIndex = static_cast<std::size_t>(
        std::distance(hostOutput.channels[0].begin(), maximum));
    EXPECT_NEAR(static_cast<double>(peakIndex), expectedIntegerDelay, 1.0);
    EXPECT_EQ(hostOutput.channels[0].size(),
              outputAdapter.totalOutputSamplesProduced());
  }
}

TEST(StreamingSampleRateAdapterTest,
     DownsamplingPreservesAudibleToneAndRejectsUltrasonicAlias) {
  constexpr double inputRate = 96000.0;
  constexpr double outputRate = kTestModelSampleRateDouble;
  constexpr std::size_t inputSamples = 96000U;
  constexpr std::size_t discard = 2048U;

  StreamingSampleRateAdapter passbandAdapter;
  ASSERT_TRUE(passbandAdapter.prepare(inputRate, outputRate, 1U));
  EXPECT_GT(passbandAdapter.filterLength(), 128U);
  EXPECT_LT(passbandAdapter.cutoffFrequencyHz(),
            0.5 * kTestModelSampleRateDouble);
  const auto passband = convertInBlocks(
      passbandAdapter, makeSine(20000.0, inputRate, inputSamples),
      {257U, 1024U, 13U});
  EXPECT_NEAR(
      measuredAmplitude(passband.channels[0], 20000.0, outputRate, discard),
      1.0, 0.002);

  StreamingSampleRateAdapter stopbandAdapter;
  ASSERT_TRUE(stopbandAdapter.prepare(inputRate, outputRate, 1U));
  const auto stopband = convertInBlocks(
      stopbandAdapter, makeSine(22500.0, inputRate, inputSamples),
      {480U, 1U, 63U});
  // 22.5 kHz is only 450 Hz beyond the new Nyquist edge and aliases to
  // 21.6 kHz if the anti-alias filter leaks.
  EXPECT_LT(
      measuredAmplitude(stopband.channels[0], 21600.0, outputRate, discard),
      3.2e-6);
}

TEST(StreamingSampleRateAdapterTest,
     UpsamplingPreservesAudibleToneAndRejectsFirstImage) {
  constexpr double inputRate = kTestModelSampleRateDouble;
  constexpr double outputRate = 96000.0;
  // Retain one exact 96,000-sample second between the discarded ends so the
  // image projection is coherent rather than limited by rectangular-window
  // leakage from the 18 kHz fundamental.
  constexpr std::size_t inputSamples =
      2U * static_cast<std::size_t>(kTestModelSampleRate);
  constexpr std::size_t discard = 48000U;
  StreamingSampleRateAdapter adapter;
  ASSERT_TRUE(adapter.prepare(inputRate, outputRate, 1U));
  const auto output =
      convertInBlocks(adapter, makeSine(18000.0, inputRate, inputSamples),
                      {1U, 511U, 32U, 1024U});
  EXPECT_NEAR(
      measuredAmplitude(output.channels[0], 18000.0, outputRate, discard), 1.0,
      0.002);
  EXPECT_LT(measuredAmplitude(output.channels[0], 26100.0, outputRate, discard),
            3.2e-6);
}
