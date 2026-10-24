#include <StemgenRT/OnnxRuntime.h>
#include <StemgenRT/MacThreadPolicy.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>

namespace {
using namespace audio_plugin;
using Clock = std::chrono::steady_clock;
using Audio = std::array<std::vector<float>, kNumChannels>;
using Stems = std::array<Audio, kNumStems>;
long setting(const char* key, long fallback) {
  const char* value = std::getenv(key);
  if (!value)
    return fallback;
  char* end{};
  const long parsed = std::strtol(value, &end, 10);
  return end != value && *end == '\0' ? parsed : -1;
}
double us(Clock::duration duration) {
  return std::chrono::duration<double, std::micro>(duration).count();
}

// Measures the complete native runtime call, including dispatch, gathering,
// input/state handling and physical residual. It is separate from plugin/DAW
// qualification.
TEST(ParallelInferenceTest, DISABLED_BenchmarkCompleteHop) {
  const auto workers = setting("STEMGENRT_PACED_INDEPENDENT_WORKERS", 1);
  const auto measured = setting("STEMGENRT_QUALIFICATION_CALLBACKS", 10000);
  ASSERT_TRUE(workers == 1 || workers == 3);
  ASSERT_GE(measured, 10000);
  ASSERT_LE(measured, 620157);
  const char* path = std::getenv("STEMGENRT_PARALLEL_NATIVE_CSV");
  ASSERT_NE(path, nullptr);
  constexpr size_t warmup = 1024;
  const size_t count = static_cast<size_t>(measured), total = count + warmup;
  std::vector<Audio> inputs(total);
  for (size_t hop = 0; hop < total; ++hop)
    for (size_t ch = 0; ch < 2; ++ch) {
      inputs[hop][ch].resize(128);
      for (size_t i = 0; i < 128; ++i) {
        const double t = static_cast<double>(hop * 128 + i) / 44100.0;
        inputs[hop][ch][i] = static_cast<float>(
            .3 * std::sin(2.0 * juce::MathConstants<double>::pi *
                          (ch == 0 ? 55 : 65) * t) +
            .25 * std::sin(2.0 * juce::MathConstants<double>::pi *
                           (ch == 0 ? 220 : 330) * t) +
            .2 * std::sin(2.0 * juce::MathConstants<double>::pi *
                          (ch == 0 ? 880 : 1320) * t));
      }
    }
  Audio aligned;
  Stems output;
  for (auto& c : aligned)
    c.resize(128);
  for (auto& s : output)
    for (auto& c : s)
      c.resize(128);
  std::vector<double> runs(count), lateness(count), completion(count);
  std::vector<int> late(count);
  OnnxRuntime runtime;
  juce::String error;
  ASSERT_TRUE(runtime.setIndependentWorkerCount(static_cast<int>(workers)));
  const auto model =
      juce::File::getSpecialLocation(juce::File::currentExecutableFile)
          .getParentDirectory()
          .getParentDirectory()
          .getChildFile("Resources/model.onnx");
  ASSERT_TRUE(runtime.loadModel(model.getFullPathName(), error, 1))
      << error.toStdString();
  ASSERT_TRUE(runtime.prepareForInference(error));
#if defined(__APPLE__)
  ASSERT_EQ(pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0), 0);
  constexpr bool realtime = true;
#else
  constexpr bool realtime = false;
#endif
  ASSERT_TRUE(runtime.startAuxiliaryWorkers(realtime, {nullptr, nullptr}));
  const juce::ScopedNoDenormals noDenormals;
  MacThreadPolicy policy;
  const bool initial = policy.start(realtime, 2700000);
#if defined(__APPLE__)
  ASSERT_TRUE(initial);
#endif
  const auto period = std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double>(128.0 / 44100.0));
  const auto epoch = Clock::now() + std::chrono::milliseconds(10);
  size_t failures = 0;
  for (size_t hop = 0; hop < total; ++hop) {
    const auto due = epoch + period * static_cast<Clock::duration::rep>(hop);
    std::this_thread::sleep_until(due);
    bool valid = false;
    const auto began = Clock::now();
    const bool ok = runtime.runInference(inputs[hop], output, aligned, valid);
    const auto ended = Clock::now();
    failures += (!ok || valid != (hop != 0)) ? 1U : 0U;
    if (hop >= warmup) {
      const size_t i = hop - warmup;
      runs[i] = us(ended - began);
      lateness[i] = us(began - due);
      completion[i] = us(ended - due);
      late[i] = ended > due + period ? 1 : 0;
    }
  }
  const bool final = policy.finish();
  runtime.stopAuxiliaryWorkers();
  std::ofstream csv(path);
  ASSERT_TRUE(csv.good());
  csv << std::setprecision(12)
      << "hop,run_wall_us,start_lateness_us,completion_from_arrival_us,"
         "deadline_miss\n";
  for (size_t i = 0; i < count; ++i)
    csv << i << ',' << runs[i] << ',' << lateness[i] << ',' << completion[i]
        << ',' << late[i] << '\n';
  csv.close();
  ASSERT_TRUE(csv.good());
  auto ordered = runs;
  std::sort(ordered.begin(), ordered.end());
  const auto quantile = [&](double q) {
    const double at = q * static_cast<double>(ordered.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(at)),
               hi = static_cast<size_t>(std::ceil(at));
    return ordered[lo] +
           (at - static_cast<double>(lo)) * (ordered[hi] - ordered[lo]);
  };
  std::cerr << std::fixed << std::setprecision(6)
            << "STEMGENRT_PARALLEL_NATIVE schema=1 workers=" << workers
            << " warmup_hops=" << warmup << " measured_hops=" << measured
            << " failures=" << failures << " deadline_misses="
            << std::accumulate(late.begin(), late.end(), 0) << " run_mean_us="
            << std::accumulate(runs.begin(), runs.end(), 0.0) /
                   static_cast<double>(count)
            << " run_p99_us=" << quantile(.99)
            << " run_max_us=" << ordered.back()
            << " policy_initial=" << (initial ? 1 : 0)
            << " policy_final=" << (final ? 1 : 0)
            << " helper_drums_initial=" << runtime.auxiliaryInitialPolicy(0)
            << " helper_vocals_initial=" << runtime.auxiliaryInitialPolicy(1)
            << " helper_drums_final=" << runtime.auxiliaryFinalPolicy(0)
            << " helper_vocals_final=" << runtime.auxiliaryFinalPolicy(1)
            << " ort_threads_per_session="
            << runtime.getConfiguredIntraOpThreadCount()
            << " runtime=" << runtime.getRuntimeVersion()
            << " plugin_callback_measured=0\n";
  EXPECT_EQ(failures, 0U);
#if defined(__APPLE__)
  EXPECT_TRUE(final);
  if (workers == 3) {
    EXPECT_EQ(runtime.auxiliaryInitialPolicy(0), 1);
    EXPECT_EQ(runtime.auxiliaryInitialPolicy(1), 1);
    EXPECT_EQ(runtime.auxiliaryFinalPolicy(0), 1);
    EXPECT_EQ(runtime.auxiliaryFinalPolicy(1), 1);
  }
#endif
}
}  // namespace
