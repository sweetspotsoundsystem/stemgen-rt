#include <StemgenRT/Constants.h>
#include <StemgenRT/ModelInputNormalizer.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {

using AudioChunk = std::array<std::vector<float>, audio_plugin::kNumChannels>;

AudioChunk makeConstantChunk(float left, float right) {
  AudioChunk chunk;
  chunk[0].assign(static_cast<size_t>(audio_plugin::kOutputChunkSize), left);
  chunk[1].assign(static_cast<size_t>(audio_plugin::kOutputChunkSize), right);
  return chunk;
}

AudioChunk makeSparseImpulseChunk(float left, float right) {
  AudioChunk chunk = makeConstantChunk(0.0f, 0.0f);
  chunk[0][17] = left;
  chunk[1][233] = right;
  return chunk;
}

AudioChunk makeAlignedChunk() {
  return makeConstantChunk(0.0f, 0.0f);
}

std::vector<float> makeHopState(float value) {
  return std::vector<float>(static_cast<size_t>(audio_plugin::kNumChannels *
                                                audio_plugin::kOutputChunkSize),
                            value);
}

std::vector<float> makeOverlapState(float value) {
  return std::vector<float>(
      static_cast<size_t>(audio_plugin::kNumStems * audio_plugin::kNumChannels *
                          audio_plugin::kAnalysisWindowSize),
      value);
}

void expectAllNear(const std::vector<float>& values,
                   float expected,
                   float tolerance = 1.0e-6f) {
  for (const float value : values) {
    EXPECT_NEAR(value, expected, tolerance);
  }
}

float maxAbsoluteValue(const std::vector<float>& values) {
  float maximum = 0.0f;
  for (const float value : values) {
    maximum = std::max(maximum, std::abs(value));
  }
  return maximum;
}

}  // namespace

TEST(ModelInputNormalizerTest,
     PeakHeadroomPreventsSparseFullScaleImpulseOverdrive) {
  audio_plugin::ModelInputNormalizer normalizer;
  normalizer.allocate();

  auto normalizedCurrent = makeHopState(0.0f);
  auto normalizedPast = makeHopState(0.0f);
  auto normalizedOverlap = makeOverlapState(0.0f);
  AudioChunk aligned = makeAlignedChunk();
  const AudioChunk impulse = makeSparseImpulseChunk(1.0f, -1.0f);
  float gain = 0.0f;

  ASSERT_TRUE(normalizer.prepare(impulse, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, gain));
  EXPECT_FLOAT_EQ(gain, 1.0f);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(normalizedCurrent),
                  audio_plugin::kModelInputPeakCeiling);
  EXPECT_FLOAT_EQ(normalizedCurrent[17], 1.0f);
  EXPECT_FLOAT_EQ(
      normalizedCurrent[static_cast<size_t>(audio_plugin::kOutputChunkSize) +
                        233U],
      -1.0f);
}

TEST(ModelInputNormalizerTest, OrdinaryAndAlreadyHotMaterialAreNotAttenuated) {
  audio_plugin::ModelInputNormalizer normalizer;
  normalizer.allocate();

  auto normalizedCurrent = makeHopState(0.0f);
  auto normalizedPast = makeHopState(0.0f);
  auto normalizedOverlap = makeOverlapState(0.0f);
  AudioChunk aligned = makeAlignedChunk();
  float gain = 0.0f;

  const AudioChunk ordinary = makeConstantChunk(0.4f, -0.4f);
  ASSERT_TRUE(normalizer.prepare(ordinary, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, gain));
  EXPECT_FLOAT_EQ(gain, 1.0f);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(normalizedCurrent), 0.4f);

  normalizer.reset();
  const AudioChunk alreadyHot = makeConstantChunk(1.2f, -1.2f);
  ASSERT_TRUE(normalizer.prepare(alreadyHot, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, gain));
  EXPECT_FLOAT_EQ(gain, 1.0f);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(normalizedCurrent), 1.2f);
}

TEST(ModelInputNormalizerTest,
     PeakLimitedGainStillMigratesPastAndOverlapStateCoherently) {
  audio_plugin::ModelInputNormalizer normalizer;
  normalizer.allocate();

  auto normalizedCurrent = makeHopState(0.0f);
  auto normalizedPast = makeHopState(0.0f);
  auto normalizedOverlap = makeOverlapState(0.0f);
  AudioChunk aligned = makeAlignedChunk();
  const AudioChunk quiet = makeConstantChunk(0.05f, -0.05f);
  float quietGain = 0.0f;
  ASSERT_TRUE(normalizer.prepare(quiet, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, quietGain));
  ASSERT_GT(quietGain, 1.0f);
  normalizer.commit(quiet, quietGain);

  std::fill(normalizedPast.begin(), normalizedPast.end(), 0.5f);
  std::fill(normalizedOverlap.begin(), normalizedOverlap.end(), 0.25f);
  const AudioChunk impulse = makeSparseImpulseChunk(0.8f, -0.8f);
  float impulseGain = 0.0f;
  ASSERT_TRUE(normalizer.prepare(impulse, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, impulseGain));
  EXPECT_NEAR(impulseGain, audio_plugin::kModelInputPeakCeiling / 0.8f,
              1.0e-6f);

  const float stateScale = impulseGain / quietGain;
  expectAllNear(normalizedPast, 0.5f * stateScale);
  expectAllNear(normalizedOverlap, 0.25f * stateScale);
  EXPECT_FLOAT_EQ(maxAbsoluteValue(normalizedCurrent),
                  audio_plugin::kModelInputPeakCeiling);
  expectAllNear(aligned[0], 0.05f, 0.0f);
  expectAllNear(aligned[1], -0.05f, 0.0f);
}

TEST(ModelInputNormalizerTest, ExactZeroHopHoldsPeakLimitedGainAndState) {
  audio_plugin::ModelInputNormalizer normalizer;
  normalizer.allocate();

  auto normalizedCurrent = makeHopState(0.0f);
  auto normalizedPast = makeHopState(0.0f);
  auto normalizedOverlap = makeOverlapState(0.0f);
  AudioChunk aligned = makeAlignedChunk();
  const AudioChunk halfScaleImpulse = makeSparseImpulseChunk(0.5f, -0.5f);
  float impulseGain = 0.0f;
  ASSERT_TRUE(normalizer.prepare(halfScaleImpulse, normalizedCurrent,
                                 normalizedPast, normalizedOverlap, aligned,
                                 impulseGain));
  ASSERT_FLOAT_EQ(impulseGain, 2.0f);
  normalizer.commit(halfScaleImpulse, impulseGain);

  std::fill(normalizedPast.begin(), normalizedPast.end(), 0.5f);
  std::fill(normalizedOverlap.begin(), normalizedOverlap.end(), 0.25f);
  const AudioChunk zero = makeConstantChunk(0.0f, 0.0f);
  float zeroGain = 0.0f;
  ASSERT_TRUE(normalizer.prepare(zero, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, zeroGain));
  EXPECT_FLOAT_EQ(zeroGain, impulseGain);
  expectAllNear(normalizedCurrent, 0.0f, 0.0f);
  expectAllNear(normalizedPast, 0.5f, 0.0f);
  expectAllNear(normalizedOverlap, 0.25f, 0.0f);
  EXPECT_FLOAT_EQ(aligned[0][17], 0.5f);
  EXPECT_FLOAT_EQ(aligned[1][233], -0.5f);
}

TEST(ModelInputNormalizerTest,
     UsesStereoRmsThenRawPastAndMigratesAmplitudeStateBothDirections) {
  audio_plugin::ModelInputNormalizer normalizer;
  normalizer.allocate();

  auto normalizedCurrent = makeHopState(0.0f);
  auto normalizedPast = makeHopState(0.5f);
  auto normalizedOverlap = makeOverlapState(0.25f);
  AudioChunk aligned = makeAlignedChunk();
  const AudioChunk first = makeConstantChunk(0.1f, 0.2f);
  float firstGain = 0.0f;

  ASSERT_TRUE(normalizer.prepare(first, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, firstGain));
  const double firstRms = std::sqrt((0.1 * 0.1 + 0.2 * 0.2) / 2.0);
  const float expectedFirstGain = static_cast<float>(
      static_cast<double>(audio_plugin::kModelInputTargetRms) / firstRms);
  EXPECT_NEAR(firstGain, expectedFirstGain, 1.0e-6f);
  expectAllNear(normalizedPast, 0.5f);
  expectAllNear(normalizedOverlap, 0.25f);
  expectAllNear(aligned[0], 0.0f, 0.0f);
  expectAllNear(aligned[1], 0.0f, 0.0f);
  expectAllNear(std::vector<float>(
                    normalizedCurrent.begin(),
                    normalizedCurrent.begin() + audio_plugin::kOutputChunkSize),
                0.1f * firstGain);
  expectAllNear(std::vector<float>(
                    normalizedCurrent.begin() + audio_plugin::kOutputChunkSize,
                    normalizedCurrent.end()),
                0.2f * firstGain);
  normalizer.commit(first, firstGain);

  // Simulate graph-owned amplitude state after the first successful run.
  std::fill(normalizedPast.begin(), normalizedPast.end(), 0.5f);
  std::fill(normalizedOverlap.begin(), normalizedOverlap.end(), 0.25f);
  const AudioChunk louder = makeConstantChunk(0.3f, 0.3f);
  float secondGain = 0.0f;
  ASSERT_TRUE(normalizer.prepare(louder, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, secondGain));
  const double secondRms =
      std::sqrt((0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.3 * 0.3) / 4.0);
  const float expectedSecondGain = static_cast<float>(
      static_cast<double>(audio_plugin::kModelInputTargetRms) / secondRms);
  EXPECT_NEAR(secondGain, expectedSecondGain, 1.0e-6f);
  const float downwardStateRatio = secondGain / firstGain;
  ASSERT_LT(downwardStateRatio, 1.0f);
  expectAllNear(normalizedPast, 0.5f * downwardStateRatio);
  expectAllNear(normalizedOverlap, 0.25f * downwardStateRatio);
  expectAllNear(aligned[0], 0.1f, 0.0f);
  expectAllNear(aligned[1], 0.2f, 0.0f);
  expectAllNear(normalizedCurrent, 0.3f * secondGain);
  normalizer.commit(louder, secondGain);

  std::fill(normalizedPast.begin(), normalizedPast.end(), 0.4f);
  std::fill(normalizedOverlap.begin(), normalizedOverlap.end(), 0.2f);
  const AudioChunk quieter = makeConstantChunk(0.01f, 0.01f);
  float thirdGain = 0.0f;
  ASSERT_TRUE(normalizer.prepare(quieter, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, thirdGain));
  ASSERT_GT(thirdGain, secondGain);
  const float upwardStateRatio = thirdGain / secondGain;
  expectAllNear(normalizedPast, 0.4f * upwardStateRatio);
  expectAllNear(normalizedOverlap, 0.2f * upwardStateRatio);
  expectAllNear(aligned[0], 0.3f, 0.0f);
  expectAllNear(aligned[1], 0.3f, 0.0f);
}

TEST(ModelInputNormalizerTest, IsBoostOnlyCapsGainAndHoldsExactZeroHop) {
  EXPECT_NEAR(20.0 * std::log10(static_cast<double>(
                         audio_plugin::kModelInputTargetRms)),
              -12.0, 1.0e-5);
  EXPECT_NEAR(
      20.0 * std::log10(static_cast<double>(audio_plugin::kModelInputMaxBoost)),
      40.0, 1.0e-5);
  EXPECT_FLOAT_EQ(audio_plugin::kModelInputPeakCeiling, 1.0f);

  audio_plugin::ModelInputNormalizer normalizer;
  normalizer.allocate();

  auto normalizedCurrent = makeHopState(0.0f);
  auto normalizedPast = makeHopState(0.5f);
  auto normalizedOverlap = makeOverlapState(0.25f);
  AudioChunk aligned = makeAlignedChunk();
  float gain = 0.0f;

  const AudioChunk initialZero = makeConstantChunk(0.0f, 0.0f);
  ASSERT_TRUE(normalizer.prepare(initialZero, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, gain));
  EXPECT_FLOAT_EQ(gain, 1.0f);

  normalizer.reset();
  const AudioChunk aboveTarget =
      makeConstantChunk(2.0f * audio_plugin::kModelInputTargetRms,
                        2.0f * audio_plugin::kModelInputTargetRms);
  ASSERT_TRUE(normalizer.prepare(aboveTarget, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, gain));
  EXPECT_FLOAT_EQ(gain, 1.0f);
  expectAllNear(normalizedCurrent, 2.0f * audio_plugin::kModelInputTargetRms);

  normalizer.reset();
  const float belowCap = audio_plugin::kModelInputTargetRms /
                         (2.0f * audio_plugin::kModelInputMaxBoost);
  const AudioChunk tiny = makeConstantChunk(belowCap, belowCap);
  ASSERT_TRUE(normalizer.prepare(tiny, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, gain));
  EXPECT_FLOAT_EQ(gain, audio_plugin::kModelInputMaxBoost);
  normalizer.commit(tiny, gain);

  std::fill(normalizedPast.begin(), normalizedPast.end(), 0.5f);
  std::fill(normalizedOverlap.begin(), normalizedOverlap.end(), 0.25f);
  ASSERT_TRUE(normalizer.prepare(initialZero, normalizedCurrent, normalizedPast,
                                 normalizedOverlap, aligned, gain));
  EXPECT_FLOAT_EQ(gain, audio_plugin::kModelInputMaxBoost);
  expectAllNear(normalizedPast, 0.5f);
  expectAllNear(normalizedOverlap, 0.25f);
  expectAllNear(aligned[0], belowCap, 0.0f);
  expectAllNear(aligned[1], belowCap, 0.0f);
}

TEST(ModelInputNormalizerTest, RejectsNonFiniteInputWithoutAdvancingRawState) {
  audio_plugin::ModelInputNormalizer normalizer;
  normalizer.allocate();

  auto normalizedCurrent = makeHopState(0.0f);
  auto normalizedPast = makeHopState(0.0f);
  auto normalizedOverlap = makeOverlapState(0.0f);
  AudioChunk aligned = makeAlignedChunk();
  float gain = 0.0f;

  for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
    AudioChunk input = makeConstantChunk(0.1f, 0.2f);
    input[1][17] = invalid;
    EXPECT_FALSE(normalizer.prepare(input, normalizedCurrent, normalizedPast,
                                    normalizedOverlap, aligned, gain));
    EXPECT_FALSE(normalizer.hasPastAudio());
  }
}
