#include <StemgenRT/PluginProcessor.h>
#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>

namespace audio_plugin_test {

// ============================================================================
// Test Fixture
// ============================================================================

class AudioProcessorTest : public ::testing::Test {
protected:
  void SetUp() override {
    processor = std::make_unique<audio_plugin::AudioPluginAudioProcessor>();
  }

  void TearDown() override {
    processor.reset();
  }

  std::unique_ptr<audio_plugin::AudioPluginAudioProcessor> processor;
};

// ============================================================================
// Basic Properties
// ============================================================================

TEST_F(AudioProcessorTest, GetName) {
  EXPECT_FALSE(processor->getName().isEmpty());
  // The name should contain "StemgenRT" or similar
  EXPECT_TRUE(processor->getName().containsIgnoreCase("Stemgen") ||
              processor->getName().containsIgnoreCase("Audio"));
}

TEST_F(AudioProcessorTest, AcceptsMidi) {
  // This plugin doesn't process MIDI
  EXPECT_FALSE(processor->acceptsMidi());
}

TEST_F(AudioProcessorTest, ProducesMidi) {
  // This plugin doesn't produce MIDI
  EXPECT_FALSE(processor->producesMidi());
}

TEST_F(AudioProcessorTest, IsMidiEffect) {
  // This is not a MIDI effect
  EXPECT_FALSE(processor->isMidiEffect());
}

TEST_F(AudioProcessorTest, GetTailLengthSeconds) {
  // No tail (reverb, delay, etc.)
  EXPECT_EQ(processor->getTailLengthSeconds(), 0.0);
}

TEST_F(AudioProcessorTest, HasEditor) {
  EXPECT_TRUE(processor->hasEditor());
}

TEST_F(AudioProcessorTest, GetNumPrograms) {
  // At least 1 program required for compatibility
  EXPECT_GE(processor->getNumPrograms(), 1);
}

TEST_F(AudioProcessorTest, GetCurrentProgram) {
  EXPECT_EQ(processor->getCurrentProgram(), 0);
}

TEST_F(AudioProcessorTest, SetCurrentProgram) {
  // Should not crash when setting program
  processor->setCurrentProgram(0);
  EXPECT_EQ(processor->getCurrentProgram(), 0);
}

TEST_F(AudioProcessorTest, GetProgramName) {
  // Program name can be empty but shouldn't crash
  juce::String name = processor->getProgramName(0);
  // Just verifying no crash - name can be empty
  (void)name;
}

TEST_F(AudioProcessorTest, ChangeProgramName) {
  // Should not crash when changing program name
  processor->changeProgramName(0, "TestProgram");
}

// ============================================================================
// Bus Layout Support
// ============================================================================

TEST_F(AudioProcessorTest, SupportsStereoInputStereoOutputs) {
  // The standard layout: stereo input, 5 stereo outputs (main + 4 stems)
  juce::AudioProcessor::BusesLayout layout;
  
  // Input: 1 stereo bus
  layout.inputBuses.add(juce::AudioChannelSet::stereo());
  
  // Output: 5 stereo buses (Main, Drums, Bass, Other, Vocals)
  for (int i = 0; i < 5; ++i) {
    layout.outputBuses.add(juce::AudioChannelSet::stereo());
  }
  
  EXPECT_TRUE(processor->isBusesLayoutSupported(layout));
}

TEST_F(AudioProcessorTest, SupportsStereoWithSomeDisabledOutputs) {
  // Layout with some stem outputs disabled
  juce::AudioProcessor::BusesLayout layout;
  
  // Input: 1 stereo bus
  layout.inputBuses.add(juce::AudioChannelSet::stereo());
  
  // Output: Main stereo + some disabled stem buses
  layout.outputBuses.add(juce::AudioChannelSet::stereo());    // Main - must be stereo
  layout.outputBuses.add(juce::AudioChannelSet::disabled());  // Drums - disabled
  layout.outputBuses.add(juce::AudioChannelSet::stereo());    // Bass - stereo
  layout.outputBuses.add(juce::AudioChannelSet::disabled());  // Other - disabled
  layout.outputBuses.add(juce::AudioChannelSet::stereo());    // Vocals - stereo
  
  EXPECT_TRUE(processor->isBusesLayoutSupported(layout));
}

TEST_F(AudioProcessorTest, RejectsMonoInput) {
  juce::AudioProcessor::BusesLayout layout;
  
  // Input: mono (should be rejected)
  layout.inputBuses.add(juce::AudioChannelSet::mono());
  
  // Output: 5 stereo buses
  for (int i = 0; i < 5; ++i) {
    layout.outputBuses.add(juce::AudioChannelSet::stereo());
  }
  
  EXPECT_FALSE(processor->isBusesLayoutSupported(layout));
}

TEST_F(AudioProcessorTest, RejectsMonoMainOutput) {
  juce::AudioProcessor::BusesLayout layout;
  
  // Input: stereo
  layout.inputBuses.add(juce::AudioChannelSet::stereo());
  
  // Output: Main is mono (should be rejected)
  layout.outputBuses.add(juce::AudioChannelSet::mono());
  for (int i = 0; i < 4; ++i) {
    layout.outputBuses.add(juce::AudioChannelSet::stereo());
  }
  
  EXPECT_FALSE(processor->isBusesLayoutSupported(layout));
}

TEST_F(AudioProcessorTest, RejectsQuadMainOutput) {
  // Test that the main output bus (bus 0) must be stereo, not quad/surround
  juce::AudioProcessor::BusesLayout layout;
  
  // Input: stereo
  layout.inputBuses.add(juce::AudioChannelSet::stereo());
  
  // Output: 5 buses, but main is quad (should be rejected)
  layout.outputBuses.add(juce::AudioChannelSet::quadraphonic());  // Main - wrong!
  for (int i = 0; i < 4; ++i) {
    layout.outputBuses.add(juce::AudioChannelSet::stereo());
  }
  
  EXPECT_FALSE(processor->isBusesLayoutSupported(layout));
}

// ============================================================================
// Latency Reporting
// ============================================================================

TEST_F(AudioProcessorTest, LatencyIsZeroWithoutModel) {
  // Without calling prepareToPlay (model not loaded), latency should be 0
  EXPECT_EQ(processor->getLatencySamples(), 0);
}

TEST_F(AudioProcessorTest, LatencyMsCalculation) {
  // Even with 0 latency samples, the calculation shouldn't crash
  double latencyMs = processor->getLatencyMs();
  EXPECT_GE(latencyMs, 0.0);
}

// ============================================================================
// Passthrough Behavior (No Model Loaded)
// ============================================================================

TEST_F(AudioProcessorTest, PassthroughWhenNoModel) {
  // When prepareToPlay hasn't been called, the model isn't loaded,
  // and the plugin should pass input audio through to outputs.
  
  // Note: We intentionally do NOT call prepareToPlay() here.
  // The AudioProcessorTest fixture only creates the processor.
  
  // Get the expected buffer channel count from the processor's bus layout.
  // This ensures we create a buffer that matches JUCE's expectations.
  const int totalInChannels = processor->getTotalNumInputChannels();
  const int totalOutChannels = processor->getTotalNumOutputChannels();
  const int totalChannels = totalInChannels + totalOutChannels;
  const int numSamples = 512;
  
  juce::AudioBuffer<float> buffer(totalChannels, numSamples);
  buffer.clear();
  
  // Store the input signal for comparison
  std::vector<float> inputL(static_cast<size_t>(numSamples));
  std::vector<float> inputR(static_cast<size_t>(numSamples));
  
  // Put a distinctive signal in input channels (channels 0 and 1)
  for (int i = 0; i < numSamples; ++i) {
    float valL = std::sin(2.0f * 3.14159f * 440.0f * static_cast<float>(i) / 44100.0f);
    float valR = valL * 0.5f;  // Different amplitude for R channel
    inputL[static_cast<size_t>(i)] = valL;
    inputR[static_cast<size_t>(i)] = valR;
    buffer.setSample(0, i, valL);
    buffer.setSample(1, i, valR);
  }
  
  juce::MidiBuffer midiBuffer;
  processor->processBlock(buffer, midiBuffer);
  
  // Without model loaded, input should be copied to all output buses.
  // Use getBusBuffer to get the correct channel mapping (same as processor does).
  const int numOutputBuses = processor->getBusCount(false /* isInput */);
  ASSERT_GT(numOutputBuses, 0) << "Expected at least one output bus";
  
  for (int busIdx = 0; busIdx < numOutputBuses; ++busIdx) {
    auto outputBus = processor->getBusBuffer(buffer, false /* isInput */, busIdx);
    const int busChannels = outputBus.getNumChannels();
    
    // Each output bus should have the input copied to it
    for (int ch = 0; ch < std::min(2, busChannels); ++ch) {
      const float* expected = (ch == 0) ? inputL.data() : inputR.data();
      const float* actual = outputBus.getReadPointer(ch);
      
      for (int i = 0; i < numSamples; ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-6f)
            << "Output bus " << busIdx << " channel " << ch 
            << " mismatch at sample " << i;
      }
    }
  }
}

// ============================================================================
// ORT Status String
// ============================================================================

TEST_F(AudioProcessorTest, OrtStatusString) {
  // Status string should indicate ORT state
  juce::String status = processor->getOrtStatusString();
  EXPECT_FALSE(status.isEmpty());
  
  // Should mention ONNX Runtime in some form
  EXPECT_TRUE(status.containsIgnoreCase("ONNX") || 
              status.containsIgnoreCase("ORT") ||
              status.containsIgnoreCase("not") ||
              status.containsIgnoreCase("model") ||
              status.containsIgnoreCase("HS-TasNet"));
}

// ============================================================================
// prepareToPlay / releaseResources Lifecycle
// ============================================================================

TEST_F(AudioProcessorTest, PrepareToPlayDoesNotCrash) {
  // Standard call with typical values
  processor->prepareToPlay(44100.0, 512);
  processor->releaseResources();
}

TEST_F(AudioProcessorTest, PrepareToPlayWithVariousSampleRates) {
  const double sampleRates[] = {22050.0, 44100.0, 48000.0, 88200.0, 96000.0, 192000.0};
  
  for (double sr : sampleRates) {
    processor->prepareToPlay(sr, 512);
    processor->releaseResources();
  }
}

TEST_F(AudioProcessorTest, PrepareToPlayWithVariousBufferSizes) {
  const int bufferSizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
  
  for (int bs : bufferSizes) {
    processor->prepareToPlay(44100.0, bs);
    processor->releaseResources();
  }
}

TEST_F(AudioProcessorTest, MultiplePrepareReleaseCycles) {
  // Simulate what a host does when changing settings
  for (int i = 0; i < 5; ++i) {
    processor->prepareToPlay(44100.0, 512);
    processor->releaseResources();
  }
}

TEST_F(AudioProcessorTest, ResetStreamingBuffersDoesNotCrash) {
  processor->prepareToPlay(44100.0, 512);
  processor->resetStreamingBuffers();
  processor->releaseResources();
}

// ============================================================================
// State Information
// ============================================================================

TEST_F(AudioProcessorTest, GetStateInformationDoesNotCrash) {
  juce::MemoryBlock destData;
  processor->getStateInformation(destData);
  // Currently a stub, just verify no crash
}

TEST_F(AudioProcessorTest, SetStateInformationDoesNotCrash) {
  // Empty data shouldn't crash
  processor->setStateInformation(nullptr, 0);
  
  // Some data shouldn't crash
  const char testData[] = "test";
  processor->setStateInformation(testData, sizeof(testData));
}

TEST_F(AudioProcessorTest, GetSetStateRoundTrip) {
  // Get state, then set it back - shouldn't crash
  juce::MemoryBlock destData;
  processor->getStateInformation(destData);
  
  if (destData.getSize() > 0) {
    processor->setStateInformation(destData.getData(), static_cast<int>(destData.getSize()));
  }
}

// ============================================================================
// processBlock Tests
// ============================================================================

class ProcessBlockTest : public AudioProcessorTest {
protected:
  void SetUp() override {
    AudioProcessorTest::SetUp();
    // Prepare the processor for playback
    processor->prepareToPlay(44100.0, 512);
  }

  void TearDown() override {
    processor->releaseResources();
    AudioProcessorTest::TearDown();
  }

  // Create a buffer with the expected bus layout
  juce::AudioBuffer<float> createBuffer(int numSamples) {
    // 2 input channels + 10 output channels (5 stereo buses)
    // JUCE packs all channels into one buffer
    return juce::AudioBuffer<float>(12, numSamples);
  }

  // Fill buffer with test signal
  void fillWithTestSignal(juce::AudioBuffer<float>& buffer, float amplitude = 0.5f) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
      for (int i = 0; i < buffer.getNumSamples(); ++i) {
        // Simple sine wave test signal
        buffer.setSample(ch, i, amplitude * std::sin(2.0f * 3.14159f * 440.0f * static_cast<float>(i) / 44100.0f));
      }
    }
  }

  // Check if buffer is silent (all zeros)
  bool isSilent(const juce::AudioBuffer<float>& buffer, int startChannel = 0, int numChannels = -1) {
    if (numChannels < 0) numChannels = buffer.getNumChannels();
    
    for (int ch = startChannel; ch < startChannel + numChannels && ch < buffer.getNumChannels(); ++ch) {
      for (int i = 0; i < buffer.getNumSamples(); ++i) {
        if (std::abs(buffer.getSample(ch, i)) > 1e-6f) {
          return false;
        }
      }
    }
    return true;
  }

  // Check if two channel ranges have the same content
  bool channelsMatch(const juce::AudioBuffer<float>& buffer, int ch1, int ch2) {
    for (int i = 0; i < buffer.getNumSamples(); ++i) {
      if (std::abs(buffer.getSample(ch1, i) - buffer.getSample(ch2, i)) > 1e-6f) {
        return false;
      }
    }
    return true;
  }
};

TEST_F(ProcessBlockTest, ProcessBlockDoesNotCrash) {
  auto buffer = createBuffer(512);
  juce::MidiBuffer midiBuffer;
  
  fillWithTestSignal(buffer);
  
  // Should not crash
  processor->processBlock(buffer, midiBuffer);
}

TEST_F(ProcessBlockTest, ProcessBlockHandlesSmallBuffers) {
  juce::MidiBuffer midiBuffer;
  
  // Test with very small buffer sizes
  const int sizes[] = {1, 2, 4, 8, 16, 32, 64};
  for (int size : sizes) {
    auto buffer = createBuffer(size);
    fillWithTestSignal(buffer);
    processor->processBlock(buffer, midiBuffer);
  }
}

TEST_F(ProcessBlockTest, ProcessBlockHandlesLargeBuffers) {
  juce::MidiBuffer midiBuffer;
  
  // Test with large buffer sizes
  const int sizes[] = {1024, 2048, 4096, 8192};
  for (int size : sizes) {
    auto buffer = createBuffer(size);
    fillWithTestSignal(buffer);
    processor->processBlock(buffer, midiBuffer);
  }
}

TEST_F(ProcessBlockTest, ProcessBlockHandlesEmptyBuffer) {
  auto buffer = createBuffer(0);
  juce::MidiBuffer midiBuffer;
  
  // Should not crash with zero samples
  processor->processBlock(buffer, midiBuffer);
}

TEST_F(ProcessBlockTest, ProducesNonSilentOutputWithLoadedModel) {
  // With the model loaded (via prepareToPlay in fixture), the plugin should
  // process audio and produce non-silent output after accounting for latency.
  
  juce::MidiBuffer midiBuffer;
  
  // Process enough blocks to fill the latency buffer and get stable output
  // The model has significant latency, so we need multiple blocks
  int warmupBlocks = 50;
  int measureBlocks = 10;
  
  float maxOutputAmplitude = 0.0f;
  
  for (int block = 0; block < warmupBlocks + measureBlocks; ++block) {
    auto buffer = createBuffer(512);
    
    // Put a distinctive signal in input channels (0-1)
    for (int i = 0; i < 512; ++i) {
      float val = std::sin(2.0f * 3.14159f * 440.0f * static_cast<float>(i) / 44100.0f);
      buffer.setSample(0, i, val);
      buffer.setSample(1, i, val * 0.5f);  // Different amplitude for R channel
    }
    
    processor->processBlock(buffer, midiBuffer);
    
    // Measure output amplitude after warmup
    if (block >= warmupBlocks) {
      // Check main output bus (channels 2-3)
      for (int ch = 2; ch <= 3; ++ch) {
        for (int i = 0; i < buffer.getNumSamples(); ++i) {
          float amp = std::abs(buffer.getSample(ch, i));
          if (amp > maxOutputAmplitude) {
            maxOutputAmplitude = amp;
          }
        }
      }
    }
  }
  
  // After warmup, we should have non-silent output from the model
  EXPECT_GT(maxOutputAmplitude, 0.01f) 
      << "Model produced silent output; max amplitude: " << maxOutputAmplitude;
}

TEST_F(ProcessBlockTest, MultipleProcessBlockCalls) {
  juce::MidiBuffer midiBuffer;
  
  // Simulate real-time streaming with many blocks
  for (int block = 0; block < 100; ++block) {
    auto buffer = createBuffer(512);
    fillWithTestSignal(buffer, 0.5f);
    processor->processBlock(buffer, midiBuffer);
  }
}

TEST_F(ProcessBlockTest, MainBusIsDryPassthroughWithLoadedModel) {
  auto buffer = createBuffer(512);
  buffer.clear();
  juce::MidiBuffer midiBuffer;

  auto inputBus = processor->getBusBuffer(buffer, true /* isInput */, 0);
  auto mainBus = processor->getBusBuffer(buffer, false /* isInput */, 0);

  std::vector<float> inputL(static_cast<size_t>(buffer.getNumSamples()));
  std::vector<float> inputR(static_cast<size_t>(buffer.getNumSamples()));
  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    float sampleL = 0.8f * std::sin(2.0f * 3.14159f * 440.0f *
                                    static_cast<float>(i) / 44100.0f);
    float sampleR = 0.5f * std::sin(2.0f * 3.14159f * 220.0f *
                                    static_cast<float>(i) / 44100.0f);
    inputBus.setSample(0, i, sampleL);
    inputBus.setSample(1, i, sampleR);
    inputL[static_cast<size_t>(i)] = sampleL;
    inputR[static_cast<size_t>(i)] = sampleR;
  }

  processor->processBlock(buffer, midiBuffer);

  const int channelsToCheck = std::min(2, mainBus.getNumChannels());
  ASSERT_EQ(channelsToCheck, 2);
  for (int i = 0; i < buffer.getNumSamples(); ++i) {
    EXPECT_NEAR(mainBus.getSample(0, i), inputL[static_cast<size_t>(i)], 1e-6f)
        << "Main L diverged from dry input at sample " << i;
    EXPECT_NEAR(mainBus.getSample(1, i), inputR[static_cast<size_t>(i)], 1e-6f)
        << "Main R diverged from dry input at sample " << i;
  }
}

TEST_F(ProcessBlockTest, ProcessBlockAfterReset) {
  juce::MidiBuffer midiBuffer;
  
  // Process some blocks
  for (int i = 0; i < 10; ++i) {
    auto buffer = createBuffer(512);
    fillWithTestSignal(buffer);
    processor->processBlock(buffer, midiBuffer);
  }
  
  // Reset
  processor->resetStreamingBuffers();
  
  // Process more blocks - should not crash
  for (int i = 0; i < 10; ++i) {
    auto buffer = createBuffer(512);
    fillWithTestSignal(buffer);
    processor->processBlock(buffer, midiBuffer);
  }
}

// ============================================================================
// Constants Validation
// ============================================================================

TEST(ConstantsTest, StemCountIsCorrect) {
  EXPECT_EQ(audio_plugin::kNumStems, 4);  // drums, bass, other, vocals
}

TEST(ConstantsTest, ChannelCountIsStereo) {
  EXPECT_EQ(audio_plugin::kNumChannels, 2);
}

TEST(ConstantsTest, ChunkSizesAreConsistent) {
  // Internal chunk should equal: context + output + context
  EXPECT_EQ(audio_plugin::kInternalChunkSize, 
            audio_plugin::kContextSize + audio_plugin::kOutputChunkSize + audio_plugin::kContextSize);
}

TEST(ConstantsTest, OutputChunkSizeIsReasonable) {
  // Should be power of 2 or at least reasonable for audio
  EXPECT_GT(audio_plugin::kOutputChunkSize, 0);
  EXPECT_LE(audio_plugin::kOutputChunkSize, 4096);
}

TEST(ConstantsTest, ContextSizeIsReasonable) {
  EXPECT_GT(audio_plugin::kContextSize, 0);
  EXPECT_LE(audio_plugin::kContextSize, 8192);
}

TEST(ConstantsTest, CrossoverFrequencyIsReasonable) {
  // Crossover should remain in a practical low-frequency range.
  EXPECT_GT(audio_plugin::kCrossoverFreqHz, 20.0f);
  EXPECT_LE(audio_plugin::kCrossoverFreqHz, 300.0f);
}

TEST(ConstantsTest, InferenceBufferCountIsPositive) {
  EXPECT_GT(audio_plugin::kNumInferenceBuffers, 0);
}

// ============================================================================
// Editor Creation
// ============================================================================

TEST_F(AudioProcessorTest, CreateEditorDoesNotCrash) {
  // Create and immediately delete the editor
  auto* editor = processor->createEditor();
  ASSERT_NE(editor, nullptr);
  delete editor;
}

TEST_F(AudioProcessorTest, MultipleEditorCreation) {
  // Some hosts create/destroy editors multiple times
  for (int i = 0; i < 3; ++i) {
    auto* editor = processor->createEditor();
    ASSERT_NE(editor, nullptr);
    delete editor;
  }
}

// ============================================================================
// Audio Quality Tests
// ============================================================================

class AudioQualityTest : public AudioProcessorTest {
protected:
  static constexpr double kSampleRate = 44100.0;
  static constexpr int kBlockSize = 512;
  static constexpr float kPi = 3.14159265358979323846f;
  
  // Random number generator for reproducible noise generation
  std::mt19937 m_randomGenerator{42};  // Seed for reproducibility
  std::uniform_real_distribution<float> m_distribution{-1.0f, 1.0f};
  
  void SetUp() override {
    AudioProcessorTest::SetUp();
    processor->prepareToPlay(kSampleRate, kBlockSize);
  }

  void TearDown() override {
    processor->releaseResources();
    AudioProcessorTest::TearDown();
  }

  // Generate a sine wave test signal
  void generateSineWave(juce::AudioBuffer<float>& buffer, float frequency, 
                        float amplitude, int startChannel = 0, int numChannels = 2) {
    for (int ch = startChannel; ch < startChannel + numChannels && ch < buffer.getNumChannels(); ++ch) {
      for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float sample = amplitude * std::sin(2.0f * kPi * frequency * static_cast<float>(i) / static_cast<float>(kSampleRate));
        buffer.setSample(ch, i, sample);
      }
    }
  }

  // Generate white noise test signal
  void generateNoise(juce::AudioBuffer<float>& buffer, float amplitude,
                     int startChannel = 0, int numChannels = 2) {
    for (int ch = startChannel; ch < startChannel + numChannels && ch < buffer.getNumChannels(); ++ch) {
      for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float noise = m_distribution(m_randomGenerator);
        buffer.setSample(ch, i, amplitude * noise);
      }
    }
  }

  // Generate impulse (click) for latency testing
  void generateImpulse(juce::AudioBuffer<float>& buffer, int samplePosition,
                       float amplitude, int startChannel = 0, int numChannels = 2) {
    buffer.clear();
    for (int ch = startChannel; ch < startChannel + numChannels && ch < buffer.getNumChannels(); ++ch) {
      if (samplePosition >= 0 && samplePosition < buffer.getNumSamples()) {
        buffer.setSample(ch, samplePosition, amplitude);
      }
    }
  }

  // Check for NaN or Inf values in buffer
  bool hasNaNOrInf(const juce::AudioBuffer<float>& buffer, 
                   int startChannel = 0, int numChannels = -1) {
    if (numChannels < 0) numChannels = buffer.getNumChannels() - startChannel;
    
    for (int ch = startChannel; ch < startChannel + numChannels && ch < buffer.getNumChannels(); ++ch) {
      for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float sample = buffer.getSample(ch, i);
        if (std::isnan(sample) || std::isinf(sample)) {
          return true;
        }
      }
    }
    return false;
  }

  // Get maximum absolute amplitude in buffer
  float getMaxAmplitude(const juce::AudioBuffer<float>& buffer,
                        int startChannel = 0, int numChannels = -1) {
    if (numChannels < 0) numChannels = buffer.getNumChannels() - startChannel;
    
    float maxAmp = 0.0f;
    for (int ch = startChannel; ch < startChannel + numChannels && ch < buffer.getNumChannels(); ++ch) {
      for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float amp = std::abs(buffer.getSample(ch, i));
        if (amp > maxAmp) maxAmp = amp;
      }
    }
    return maxAmp;
  }

  // Calculate RMS energy of buffer
  float calculateRMS(const juce::AudioBuffer<float>& buffer,
                     int startChannel = 0, int numChannels = -1) {
    if (numChannels < 0) numChannels = buffer.getNumChannels() - startChannel;
    
    float sumSquares = 0.0f;
    int totalSamples = 0;
    
    for (int ch = startChannel; ch < startChannel + numChannels && ch < buffer.getNumChannels(); ++ch) {
      for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float sample = buffer.getSample(ch, i);
        sumSquares += sample * sample;
        totalSamples++;
      }
    }
    
    return totalSamples > 0 ? std::sqrt(sumSquares / static_cast<float>(totalSamples)) : 0.0f;
  }

  // Check if buffer is silent (below threshold)
  bool isSilent(const juce::AudioBuffer<float>& buffer, float threshold = 1e-6f,
                int startChannel = 0, int numChannels = -1) {
    return getMaxAmplitude(buffer, startChannel, numChannels) < threshold;
  }

  // Calculate correlation between two channels
  float calculateCorrelation(const juce::AudioBuffer<float>& buffer, int ch1, int ch2) {
    if (ch1 >= buffer.getNumChannels() || ch2 >= buffer.getNumChannels()) return 0.0f;
    
    float sum1 = 0.0f, sum2 = 0.0f, sumProduct = 0.0f;
    float sumSq1 = 0.0f, sumSq2 = 0.0f;
    int n = buffer.getNumSamples();
    
    for (int i = 0; i < n; ++i) {
      float s1 = buffer.getSample(ch1, i);
      float s2 = buffer.getSample(ch2, i);
      sum1 += s1;
      sum2 += s2;
      sumProduct += s1 * s2;
      sumSq1 += s1 * s1;
      sumSq2 += s2 * s2;
    }
    
    float nf = static_cast<float>(n);
    float numerator = nf * sumProduct - sum1 * sum2;
    float denominator = std::sqrt((nf * sumSq1 - sum1 * sum1) * (nf * sumSq2 - sum2 * sum2));
    
    return denominator > 1e-10f ? numerator / denominator : 0.0f;
  }

  // Check if model is loaded (latency > 0 indicates model is active)
  bool isModelLoaded() {
    return processor->getLatencySamples() > 0;
  }

  // Iterate stem output samples using JUCE bus mapping (buses 1..4).
  template <typename Fn>
  void forEachStemOutputSample(juce::AudioBuffer<float>& buffer, Fn&& fn) {
    const int numOutputBuses = processor->getBusCount(false /* isInput */);
    for (int busIdx = 1; busIdx < numOutputBuses; ++busIdx) {
      auto stemBus = processor->getBusBuffer(buffer, false /* isInput */, busIdx);
      for (int ch = 0; ch < stemBus.getNumChannels(); ++ch) {
        const float* readPtr = stemBus.getReadPointer(ch);
        for (int i = 0; i < stemBus.getNumSamples(); ++i) {
          fn(readPtr[i]);
        }
      }
    }
  }

  bool stemOutputsHaveNaNOrInf(juce::AudioBuffer<float>& buffer) {
    bool hasInvalid = false;
    forEachStemOutputSample(buffer, [&hasInvalid](float sample) {
      if (std::isnan(sample) || std::isinf(sample)) {
        hasInvalid = true;
      }
    });
    return hasInvalid;
  }

  float getStemOutputsMaxAmplitude(juce::AudioBuffer<float>& buffer) {
    float maxAmp = 0.0f;
    forEachStemOutputSample(buffer, [&maxAmp](float sample) {
      maxAmp = std::max(maxAmp, std::abs(sample));
    });
    return maxAmp;
  }
};

// --------------------------------------------------------------------------
// Output Sanity Tests (no NaN, Inf, or extreme values)
// --------------------------------------------------------------------------

TEST_F(AudioQualityTest, OutputHasNoNaNWithSineInput) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  generateSineWave(buffer, 440.0f, 0.5f, 0, 2);  // Input channels
  processor->processBlock(buffer, midiBuffer);
  
  EXPECT_FALSE(stemOutputsHaveNaNOrInf(buffer))
      << "Stem output contains NaN or Inf values";
}

TEST_F(AudioQualityTest, OutputHasNoNaNWithNoiseInput) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  generateNoise(buffer, 0.5f, 0, 2);  // Input channels
  processor->processBlock(buffer, midiBuffer);
  
  EXPECT_FALSE(stemOutputsHaveNaNOrInf(buffer))
      << "Stem output contains NaN or Inf values with noise input";
}

TEST_F(AudioQualityTest, OutputHasNoNaNWithSilentInput) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  buffer.clear();
  juce::MidiBuffer midiBuffer;
  
  processor->processBlock(buffer, midiBuffer);
  
  EXPECT_FALSE(stemOutputsHaveNaNOrInf(buffer))
      << "Stem output contains NaN or Inf with silent input";
}

TEST_F(AudioQualityTest, OutputHasNoNaNWithImpulse) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  generateImpulse(buffer, 100, 1.0f, 0, 2);  // Full-scale impulse
  processor->processBlock(buffer, midiBuffer);
  
  EXPECT_FALSE(stemOutputsHaveNaNOrInf(buffer))
      << "Stem output contains NaN or Inf with impulse input";
}

TEST_F(AudioQualityTest, OutputHasNoNaNAfterManyBlocks) {
  juce::MidiBuffer midiBuffer;
  
  // Process many blocks to check for numerical instability over time
  for (int block = 0; block < 1000; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    generateSineWave(buffer, 440.0f, 0.8f, 0, 2);
    processor->processBlock(buffer, midiBuffer);
    
    if (stemOutputsHaveNaNOrInf(buffer)) {
      FAIL() << "Output contains NaN or Inf at block " << block;
    }
  }
}

// --------------------------------------------------------------------------
// Amplitude Bounds Tests
// --------------------------------------------------------------------------

TEST_F(AudioQualityTest, OutputAmplitudeIsReasonable) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  // Input at -6dB (0.5 amplitude)
  generateSineWave(buffer, 440.0f, 0.5f, 0, 2);
  processor->processBlock(buffer, midiBuffer);
  
  // Output should not exceed reasonable bounds (allow some headroom for processing)
  float maxOut = getStemOutputsMaxAmplitude(buffer);
  EXPECT_LE(maxOut, 2.0f) << "Output amplitude " << maxOut << " exceeds reasonable bounds";
}

TEST_F(AudioQualityTest, FullScaleInputProducesReasonableOutput) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  // Full-scale input (0 dBFS)
  generateSineWave(buffer, 440.0f, 1.0f, 0, 2);
  processor->processBlock(buffer, midiBuffer);
  
  // Output shouldn't explode even with full-scale input
  float maxOut = getStemOutputsMaxAmplitude(buffer);
  EXPECT_LE(maxOut, 4.0f) << "Full-scale input caused output explosion: " << maxOut;
}

TEST_F(AudioQualityTest, HotInputDoesNotCauseClipping) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  // Hot input (+6dB over full scale - simulating DAW clipping scenarios)
  generateSineWave(buffer, 440.0f, 2.0f, 0, 2);
  processor->processBlock(buffer, midiBuffer);
  
  // Should not produce inf/nan even with hot input
  EXPECT_FALSE(stemOutputsHaveNaNOrInf(buffer))
      << "Hot input caused numerical issues";
}

// --------------------------------------------------------------------------
// Silent Input/Output Tests
// --------------------------------------------------------------------------

TEST_F(AudioQualityTest, SilentInputProducesSilentOutput) {
  juce::MidiBuffer midiBuffer;
  
  // Process several blocks of silence to flush any initial transients
  for (int block = 0; block < 20; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    buffer.clear();
    processor->processBlock(buffer, midiBuffer);
  }
  
  // Now check if output is silent
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  buffer.clear();
  processor->processBlock(buffer, midiBuffer);
  
  // Output should be silent (or nearly silent - threshold for numerical noise)
  float maxOut = getStemOutputsMaxAmplitude(buffer);
  EXPECT_LT(maxOut, 1e-4f) << "Silent input produced non-silent output: " << maxOut;
}

TEST_F(AudioQualityTest, TransitionToSilenceDecaysCleanly) {
  juce::MidiBuffer midiBuffer;
  
  // First, feed some audio
  for (int block = 0; block < 10; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    generateSineWave(buffer, 440.0f, 0.5f, 0, 2);
    processor->processBlock(buffer, midiBuffer);
  }
  
  // Then transition to silence and track decay
  for (int block = 0; block < 50; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    buffer.clear();
    processor->processBlock(buffer, midiBuffer);
    
    float currentMax = getStemOutputsMaxAmplitude(buffer);
    
    // After some blocks, should be essentially silent
    if (block > 20) {
      EXPECT_LT(currentMax, 0.01f) << "Output didn't decay to silence by block " << block;
    }
  }
}

// --------------------------------------------------------------------------
// Stereo Coherence Tests
// --------------------------------------------------------------------------

TEST_F(AudioQualityTest, MonoInputProducesCoherentStereoOutput) {
  juce::MidiBuffer midiBuffer;
  
  // Need to process multiple blocks for the system to stabilize
  // (model has latency, output ring buffer needs to fill)
  int warmupBlocks = 30;  // Allow time for latency + buffer fill
  int measureBlocks = 10;
  
  float totalCorrelation = 0.0f;
  int measurementCount = 0;
  
  for (int block = 0; block < warmupBlocks + measureBlocks; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    
    // Generate identical signal on both input channels (mono)
    // Use block offset to create continuous signal
    for (int i = 0; i < kBlockSize; ++i) {
      int sampleIdx = block * kBlockSize + i;
      float sample = 0.5f * std::sin(2.0f * kPi * 440.0f * static_cast<float>(sampleIdx) / static_cast<float>(kSampleRate));
      buffer.setSample(0, i, sample);
      buffer.setSample(1, i, sample);
    }
    
    processor->processBlock(buffer, midiBuffer);
    
    // Only measure after warmup
    if (block >= warmupBlocks) {
      float correlation = calculateCorrelation(buffer, 2, 3);
      // Only count if there's actual signal (not silent output)
      float maxAmp = getMaxAmplitude(buffer, 2, 2);
      if (maxAmp > 0.01f) {
        totalCorrelation += correlation;
        measurementCount++;
      }
    }
  }
  
  // Main output L/R should be highly correlated (coherent stereo)
  if (measurementCount > 0) {
    float avgCorrelation = totalCorrelation / static_cast<float>(measurementCount);
    EXPECT_GT(avgCorrelation, 0.9f) << "Mono input produced incoherent stereo output, avg correlation: " << avgCorrelation;
  } else {
    // If no signal was produced, that's also a problem worth noting
    ADD_FAILURE() << "No output signal detected after warmup";
  }
}

TEST_F(AudioQualityTest, StereoImageIsPreserved) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  // Generate stereo signal with different content L/R
  for (int i = 0; i < kBlockSize; ++i) {
    buffer.setSample(0, i, 0.5f * std::sin(2.0f * kPi * 440.0f * static_cast<float>(i) / static_cast<float>(kSampleRate)));   // L: 440 Hz
    buffer.setSample(1, i, 0.5f * std::sin(2.0f * kPi * 880.0f * static_cast<float>(i) / static_cast<float>(kSampleRate)));   // R: 880 Hz
  }
  
  float inputCorrelation = calculateCorrelation(buffer, 0, 1);
  
  processor->processBlock(buffer, midiBuffer);
  
  float outputCorrelation = calculateCorrelation(buffer, 2, 3);
  
  // Output correlation should be similar to input correlation
  // (stereo image preserved, not collapsed or inverted)
  float correlationDiff = std::abs(outputCorrelation - inputCorrelation);
  EXPECT_LT(correlationDiff, 0.5f) << "Stereo image changed significantly. Input corr: " 
                                    << inputCorrelation << ", Output corr: " << outputCorrelation;
}

// --------------------------------------------------------------------------
// Energy Conservation Tests
// --------------------------------------------------------------------------

TEST_F(AudioQualityTest, EnergyIsRoughlyPreserved) {
  juce::MidiBuffer midiBuffer;
  
  // Need to process enough blocks for the system to stabilize
  float inputEnergySum = 0.0f;
  float outputEnergySum = 0.0f;
  int measurementBlocks = 50;
  int warmupBlocks = 20;
  
  for (int block = 0; block < warmupBlocks + measurementBlocks; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    generateSineWave(buffer, 440.0f, 0.5f, 0, 2);
    
    // Measure input energy (only during measurement phase)
    if (block >= warmupBlocks) {
      inputEnergySum += calculateRMS(buffer, 0, 2);
    }
    
    processor->processBlock(buffer, midiBuffer);
    
    // Measure main output energy (only during measurement phase)
    if (block >= warmupBlocks) {
      outputEnergySum += calculateRMS(buffer, 2, 2);
    }
  }
  
  float avgInputEnergy = inputEnergySum / static_cast<float>(measurementBlocks);
  float avgOutputEnergy = outputEnergySum / static_cast<float>(measurementBlocks);
  
  // Energy ratio should be close to 1.0 (within reasonable tolerance)
  // Allow wide tolerance since separation can redistribute energy
  if (avgInputEnergy > 0.01f) {
    float energyRatio = avgOutputEnergy / avgInputEnergy;
    EXPECT_GT(energyRatio, 0.1f) << "Output energy too low: " << avgOutputEnergy << " vs input: " << avgInputEnergy;
    EXPECT_LT(energyRatio, 10.0f) << "Output energy too high: " << avgOutputEnergy << " vs input: " << avgInputEnergy;
  }
}

// --------------------------------------------------------------------------
// Frequency Response Tests
// --------------------------------------------------------------------------

TEST_F(AudioQualityTest, LowFrequencyPassesThrough) {
  juce::MidiBuffer midiBuffer;
  
  // Process several blocks of low frequency content (below crossover)
  float totalInputEnergy = 0.0f;
  float totalOutputEnergy = 0.0f;
  
  for (int block = 0; block < 100; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    generateSineWave(buffer, 50.0f, 0.5f, 0, 2);  // 50 Hz - below 80 Hz crossover
    
    if (block >= 20) {  // Skip warmup
      totalInputEnergy += calculateRMS(buffer, 0, 2);
    }
    
    processor->processBlock(buffer, midiBuffer);
    
    if (block >= 20) {
      totalOutputEnergy += calculateRMS(buffer, 2, 2);
    }
  }
  
  // Low frequencies should pass through (routed to bass stem, then to main)
  if (totalInputEnergy > 0.01f) {
    float ratio = totalOutputEnergy / totalInputEnergy;
    EXPECT_GT(ratio, 0.1f) << "Low frequency content severely attenuated";
  }
}

TEST_F(AudioQualityTest, HighFrequencyPassesThrough) {
  juce::MidiBuffer midiBuffer;
  
  // Process high frequency content (well above crossover)
  float totalInputEnergy = 0.0f;
  float totalOutputEnergy = 0.0f;
  
  for (int block = 0; block < 100; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    generateSineWave(buffer, 4000.0f, 0.5f, 0, 2);  // 4 kHz - well above crossover
    
    if (block >= 20) {
      totalInputEnergy += calculateRMS(buffer, 0, 2);
    }
    
    processor->processBlock(buffer, midiBuffer);
    
    if (block >= 20) {
      totalOutputEnergy += calculateRMS(buffer, 2, 2);
    }
  }
  
  // High frequencies should pass through
  if (totalInputEnergy > 0.01f) {
    float ratio = totalOutputEnergy / totalInputEnergy;
    EXPECT_GT(ratio, 0.1f) << "High frequency content severely attenuated";
  }
}

// --------------------------------------------------------------------------
// Latency Tests
// --------------------------------------------------------------------------

TEST_F(AudioQualityTest, ReportedLatencyIsConsistent) {
  int latency1 = processor->getLatencySamples();
  
  // Process some blocks
  juce::MidiBuffer midiBuffer;
  for (int i = 0; i < 10; ++i) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    generateSineWave(buffer, 440.0f, 0.5f, 0, 2);
    processor->processBlock(buffer, midiBuffer);
  }
  
  int latency2 = processor->getLatencySamples();
  
  // Latency should remain constant during processing
  EXPECT_EQ(latency1, latency2) << "Latency changed during processing";
}

TEST_F(AudioQualityTest, LatencyMsMatchesLatencySamples) {
  int latencySamples = processor->getLatencySamples();
  double latencyMs = processor->getLatencyMs();
  
  // Calculate expected latency in ms
  double expectedMs = (static_cast<double>(latencySamples) / kSampleRate) * 1000.0;
  
  // Should match within floating point tolerance
  EXPECT_NEAR(latencyMs, expectedMs, 0.1) << "Latency Ms doesn't match samples";
}

// --------------------------------------------------------------------------
// DC Offset Tests
// --------------------------------------------------------------------------

TEST_F(AudioQualityTest, NoDCOffsetWithACInput) {
  juce::MidiBuffer midiBuffer;
  
  // Process many blocks and accumulate DC
  double dcSum = 0.0;
  int totalSamples = 0;
  
  for (int block = 0; block < 100; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    generateSineWave(buffer, 440.0f, 0.5f, 0, 2);
    processor->processBlock(buffer, midiBuffer);
    
    // Accumulate samples from main output
    for (int ch = 2; ch <= 3; ++ch) {
      for (int i = 0; i < kBlockSize; ++i) {
        dcSum += static_cast<double>(buffer.getSample(ch, i));
        totalSamples++;
      }
    }
  }
  
  double dcOffset = dcSum / static_cast<double>(totalSamples);
  
  // DC offset should be negligible
  EXPECT_LT(std::abs(dcOffset), 0.01) << "DC offset detected: " << dcOffset;
}

TEST_F(AudioQualityTest, NoDCOffsetWithSilentInput) {
  juce::MidiBuffer midiBuffer;
  
  double dcSum = 0.0;
  int totalSamples = 0;
  
  for (int block = 0; block < 50; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    buffer.clear();
    processor->processBlock(buffer, midiBuffer);
    
    for (int ch = 2; ch <= 3; ++ch) {
      for (int i = 0; i < kBlockSize; ++i) {
        dcSum += static_cast<double>(buffer.getSample(ch, i));
        totalSamples++;
      }
    }
  }
  
  double dcOffset = dcSum / static_cast<double>(totalSamples);
  EXPECT_LT(std::abs(dcOffset), 1e-6) << "DC offset with silent input: " << dcOffset;
}

// --------------------------------------------------------------------------
// Stress Tests
// --------------------------------------------------------------------------

TEST_F(AudioQualityTest, StableUnderRapidInputChanges) {
  juce::MidiBuffer midiBuffer;
  
  for (int block = 0; block < 100; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    
    // Alternate between different signals rapidly
    switch (block % 4) {
      case 0: generateSineWave(buffer, 440.0f, 0.8f, 0, 2); break;
      case 1: buffer.clear(); break;
      case 2: generateNoise(buffer, 0.5f, 0, 2); break;
      case 3: generateImpulse(buffer, kBlockSize / 2, 1.0f, 0, 2); break;
    }
    
    processor->processBlock(buffer, midiBuffer);
    
    EXPECT_FALSE(stemOutputsHaveNaNOrInf(buffer))
        << "Numerical issues at block " << block;
  }
}

TEST_F(AudioQualityTest, StableWithIntermittentResets) {
  juce::MidiBuffer midiBuffer;
  
  for (int cycle = 0; cycle < 10; ++cycle) {
    // Process some blocks
    for (int block = 0; block < 20; ++block) {
      juce::AudioBuffer<float> buffer(12, kBlockSize);
      generateSineWave(buffer, 440.0f, 0.5f, 0, 2);
      processor->processBlock(buffer, midiBuffer);
      
      EXPECT_FALSE(stemOutputsHaveNaNOrInf(buffer))
          << "NaN/Inf after reset cycle " << cycle << " block " << block;
    }
    
    // Reset buffers (simulating transport stop/start)
    processor->resetStreamingBuffers();
  }
}

// ============================================================================
// Bass Diagnostic Tests
// ============================================================================

class BassDiagnosticTest : public AudioQualityTest {
protected:
  // Per-stem metrics for a single frequency
  struct StemMetrics {
    float rms = 0.0f;
    float peak = 0.0f;
    float crestFactor = 0.0f;  // peak/RMS — sqrt(2) for pure sine
  };

  struct FrequencyReport {
    float frequency;
    float inputRms;
    float inputPeak;
    float mainRms;
    float mainPeak;
    float energyRatio;  // main RMS / input RMS
    float stemSumRms;   // RMS of sum-of-stem-buses
    StemMetrics drums;
    StemMetrics bass;
    StemMetrics other;
    StemMetrics vocals;
  };

  // Calculate metrics for a channel range over accumulated data
  StemMetrics calcStemMetrics(const std::vector<float>& accumulatedL,
                              const std::vector<float>& accumulatedR) {
    StemMetrics m;
    float sumSq = 0.0f;
    float peak = 0.0f;
    size_t n = accumulatedL.size();
    for (size_t i = 0; i < n; ++i) {
      float sL = accumulatedL[i];
      float sR = accumulatedR[i];
      sumSq += sL * sL + sR * sR;
      peak = std::max(peak, std::max(std::abs(sL), std::abs(sR)));
    }
    m.rms = std::sqrt(sumSq / static_cast<float>(2 * n));
    m.peak = peak;
    m.crestFactor = (m.rms > 1e-10f) ? (m.peak / m.rms) : 0.0f;
    return m;
  }

  FrequencyReport runFrequencyTest(float freq, float amplitude = 0.5f) {
    // Reset processor for clean test
    processor->releaseResources();
    processor->prepareToPlay(kSampleRate, kBlockSize);

    constexpr int kWarmupBlocks = 30;
    constexpr int kMeasureBlocks = 70;
    juce::MidiBuffer midiBuffer;

    // Accumulate samples during measurement phase
    std::vector<float> inputL, inputR;
    std::vector<float> mainL, mainR;
    std::vector<float> drumsL, drumsR;
    std::vector<float> bassL, bassR;
    std::vector<float> otherL, otherR;
    std::vector<float> vocalsL, vocalsR;
    std::vector<float> stemSumL, stemSumR;

    size_t reserveSize = static_cast<size_t>(kMeasureBlocks * kBlockSize);
    inputL.reserve(reserveSize);  inputR.reserve(reserveSize);
    mainL.reserve(reserveSize);   mainR.reserve(reserveSize);
    drumsL.reserve(reserveSize);  drumsR.reserve(reserveSize);
    bassL.reserve(reserveSize);   bassR.reserve(reserveSize);
    otherL.reserve(reserveSize);  otherR.reserve(reserveSize);
    vocalsL.reserve(reserveSize); vocalsR.reserve(reserveSize);
    stemSumL.reserve(reserveSize); stemSumR.reserve(reserveSize);

    // Use 10-channel buffer (JUCE shares input ch 0-1 with output bus 0)
    const int totalChannels = processor->getTotalNumInputChannels() + processor->getTotalNumOutputChannels();

    for (int block = 0; block < kWarmupBlocks + kMeasureBlocks; ++block) {
      juce::AudioBuffer<float> buffer(totalChannels, kBlockSize);
      buffer.clear();

      // Write input via getBusBuffer for correct channel mapping
      auto inBus = processor->getBusBuffer(buffer, true, 0);
      for (int i = 0; i < kBlockSize; ++i) {
        int sampleIdx = block * kBlockSize + i;
        float sample = amplitude * std::sin(2.0f * kPi * freq *
            static_cast<float>(sampleIdx) / static_cast<float>(kSampleRate));
        inBus.setSample(0, i, sample);
        inBus.setSample(1, i, sample);
      }

      // Save input before processing (input bus shares channels with main output)
      std::vector<float> savedInputL, savedInputR;
      if (block >= kWarmupBlocks) {
        savedInputL.resize(static_cast<size_t>(kBlockSize));
        savedInputR.resize(static_cast<size_t>(kBlockSize));
        for (int i = 0; i < kBlockSize; ++i) {
          savedInputL[static_cast<size_t>(i)] = inBus.getSample(0, i);
          savedInputR[static_cast<size_t>(i)] = inBus.getSample(1, i);
        }
      }

      processor->processBlock(buffer, midiBuffer);

      // Give inference thread time to process (each 512-sample block = ~11.6ms at 44.1kHz)
      std::this_thread::sleep_for(std::chrono::milliseconds(15));

      // Collect output during measurement phase using getBusBuffer
      if (block >= kWarmupBlocks) {
        auto mainBus = processor->getBusBuffer(buffer, false, 0);
        auto drumsBus = processor->getBusBuffer(buffer, false, 1);
        auto bassBus = processor->getBusBuffer(buffer, false, 2);
        auto otherBus = processor->getBusBuffer(buffer, false, 3);
        auto vocalsBus = processor->getBusBuffer(buffer, false, 4);

        for (int i = 0; i < kBlockSize; ++i) {
          inputL.push_back(savedInputL[static_cast<size_t>(i)]);
          inputR.push_back(savedInputR[static_cast<size_t>(i)]);
          mainL.push_back(mainBus.getSample(0, i));
          mainR.push_back(mainBus.getSample(1, i));
          float dL = drumsBus.getSample(0, i);
          float dR = drumsBus.getSample(1, i);
          float bL = bassBus.getSample(0, i);
          float bR = bassBus.getSample(1, i);
          float oL = otherBus.getSample(0, i);
          float oR = otherBus.getSample(1, i);
          float vL = vocalsBus.getSample(0, i);
          float vR = vocalsBus.getSample(1, i);
          drumsL.push_back(dL);   drumsR.push_back(dR);
          bassL.push_back(bL);    bassR.push_back(bR);
          otherL.push_back(oL);   otherR.push_back(oR);
          vocalsL.push_back(vL);  vocalsR.push_back(vR);
          stemSumL.push_back(dL + bL + oL + vL);
          stemSumR.push_back(dR + bR + oR + vR);
        }
      }
    }

    // Build report
    FrequencyReport r;
    r.frequency = freq;

    // Input metrics
    auto inputM = calcStemMetrics(inputL, inputR);
    r.inputRms = inputM.rms;
    r.inputPeak = inputM.peak;

    // Main bus metrics
    auto mainM = calcStemMetrics(mainL, mainR);
    r.mainRms = mainM.rms;
    r.mainPeak = mainM.peak;

    // Energy ratio
    r.energyRatio = (r.inputRms > 1e-10f) ? (r.mainRms / r.inputRms) : 0.0f;

    // Sum-of-stems metrics
    auto sumM = calcStemMetrics(stemSumL, stemSumR);
    r.stemSumRms = sumM.rms;

    // Per-stem metrics
    r.drums = calcStemMetrics(drumsL, drumsR);
    r.bass = calcStemMetrics(bassL, bassR);
    r.other = calcStemMetrics(otherL, otherR);
    r.vocals = calcStemMetrics(vocalsL, vocalsR);

    return r;
  }

  void printReport(const std::vector<FrequencyReport>& reports) {
    std::cerr << "\n";
    std::cerr << "====================================================================\n";
    std::cerr << "  BASS DIAGNOSTIC REPORT\n";
    std::cerr << "====================================================================\n";
    std::cerr << "  Crossover: " << audio_plugin::kCrossoverFreqHz << " Hz\n";
    std::cerr << "  Norm target: " << audio_plugin::kNormTargetRmsDb << " dB\n";
    std::cerr << "  Block size: " << kBlockSize << ", Sample rate: " << kSampleRate << "\n";
    std::cerr << "====================================================================\n\n";

    // Header
    fprintf(stderr, "%-6s | %-10s %-10s | %-10s %-10s %-10s | %-10s %-10s %-10s | %-10s %-10s %-10s | %-10s %-10s %-10s | %-10s %-10s %-10s\n",
            "Freq", "InRMS", "InPeak",
            "MainRMS", "MainPeak", "E.Ratio",
            "DruRMS", "DruPeak", "DruCF",
            "BasRMS", "BasPeak", "BasCF",
            "OthRMS", "OthPeak", "OthCF",
            "VocRMS", "VocPeak", "VocCF");
    fprintf(stderr, "%-6s-+-%-10s-%-10s-+-%-10s-%-10s-%-10s-+-%-10s-%-10s-%-10s-+-%-10s-%-10s-%-10s-+-%-10s-%-10s-%-10s-+-%-10s-%-10s-%-10s\n",
            "------", "----------", "----------",
            "----------", "----------", "----------",
            "----------", "----------", "----------",
            "----------", "----------", "----------",
            "----------", "----------", "----------",
            "----------", "----------", "----------");

    for (const auto& r : reports) {
      fprintf(stderr, "%-6.0f | %-10.6f %-10.6f | %-10.6f %-10.6f %-10.4f | %-10.6f %-10.6f %-10.4f | %-10.6f %-10.6f %-10.4f | %-10.6f %-10.6f %-10.4f | %-10.6f %-10.6f %-10.4f\n",
              static_cast<double>(r.frequency), static_cast<double>(r.inputRms), static_cast<double>(r.inputPeak),
              static_cast<double>(r.mainRms), static_cast<double>(r.mainPeak), static_cast<double>(r.energyRatio),
              static_cast<double>(r.drums.rms), static_cast<double>(r.drums.peak), static_cast<double>(r.drums.crestFactor),
              static_cast<double>(r.bass.rms), static_cast<double>(r.bass.peak), static_cast<double>(r.bass.crestFactor),
              static_cast<double>(r.other.rms), static_cast<double>(r.other.peak), static_cast<double>(r.other.crestFactor),
              static_cast<double>(r.vocals.rms), static_cast<double>(r.vocals.peak), static_cast<double>(r.vocals.crestFactor));
    }

    // Consistency check: sum-of-stems vs main
    std::cerr << "\n--- Stem Sum vs Main Consistency ---\n";
    for (const auto& r : reports) {
      float diff = std::abs(r.stemSumRms - r.mainRms);
      float relDiff = (r.mainRms > 1e-10f) ? (diff / r.mainRms) * 100.0f : 0.0f;
      fprintf(stderr, "  %4.0f Hz: stemSum RMS=%.6f  main RMS=%.6f  diff=%.6f (%.2f%%)\n",
              static_cast<double>(r.frequency), static_cast<double>(r.stemSumRms),
              static_cast<double>(r.mainRms), static_cast<double>(diff), static_cast<double>(relDiff));
    }

    // Stem energy distribution
    std::cerr << "\n--- Stem Energy Distribution (% of total stem RMS) ---\n";
    for (const auto& r : reports) {
      float total = r.drums.rms + r.bass.rms + r.other.rms + r.vocals.rms;
      if (total > 1e-10f) {
        fprintf(stderr, "  %4.0f Hz: Drums=%.1f%%  Bass=%.1f%%  Other=%.1f%%  Vocals=%.1f%%\n",
                static_cast<double>(r.frequency),
                static_cast<double>(r.drums.rms / total * 100.0f),
                static_cast<double>(r.bass.rms / total * 100.0f),
                static_cast<double>(r.other.rms / total * 100.0f),
                static_cast<double>(r.vocals.rms / total * 100.0f));
      }
    }

    // Crest factor analysis
    std::cerr << "\n--- Crest Factor Analysis (pure sine = " << std::sqrt(2.0f) << ") ---\n";
    for (const auto& r : reports) {
      fprintf(stderr, "  %4.0f Hz: Drums=%.3f  Bass=%.3f  Other=%.3f  Vocals=%.3f  Main=%.3f\n",
              static_cast<double>(r.frequency), static_cast<double>(r.drums.crestFactor),
              static_cast<double>(r.bass.crestFactor), static_cast<double>(r.other.crestFactor),
              static_cast<double>(r.vocals.crestFactor),
              static_cast<double>((r.mainRms > 1e-10f) ? r.mainPeak / r.mainRms : 0.0f));
    }

    std::cerr << "\n====================================================================\n\n";
  }
};

TEST_F(BassDiagnosticTest, DiagnoseBassSaturation) {
  if (!isModelLoaded()) {
    GTEST_SKIP() << "Model not loaded, skipping diagnostic test";
  }

  const float testFreqs[] = {40.0f, 100.0f, 200.0f, 1000.0f};
  std::vector<FrequencyReport> reports;

  for (float freq : testFreqs) {
    reports.push_back(runFrequencyTest(freq, 0.5f));
  }

  printReport(reports);

  // Assertions
  for (const auto& r : reports) {
    EXPECT_TRUE(std::isfinite(r.mainRms)) << r.frequency << " Hz: main RMS is not finite";
    EXPECT_TRUE(std::isfinite(r.mainPeak)) << r.frequency << " Hz: main peak is not finite";

    // Energy ratio between 0.5 and 2.0
    EXPECT_GE(r.energyRatio, 0.5f)
        << r.frequency << " Hz: energy ratio " << r.energyRatio << " too low";
    EXPECT_LE(r.energyRatio, 2.0f)
        << r.frequency << " Hz: energy ratio " << r.energyRatio << " too high";
  }
}

TEST_F(BassDiagnosticTest, DiagnoseKickDrum) {
  if (!isModelLoaded()) {
    GTEST_SKIP() << "Model not loaded, skipping diagnostic test";
  }

  // Synthesize a realistic kick drum pattern:
  //   - Sharp transient click (broadband, <1ms)
  //   - Pitch sweep from ~200Hz down to ~50Hz (body)
  //   - Exponential amplitude decay (~150ms)
  //   - Repeating every ~500ms (120 BPM)
  constexpr float kickAmplitude = 0.8f;
  constexpr float kickStartFreq = 200.0f;
  constexpr float kickEndFreq = 50.0f;
  constexpr float kickDecayMs = 150.0f;
  constexpr float kickIntervalMs = 500.0f;  // 120 BPM

  const float kickDecaySamples = kickDecayMs * static_cast<float>(kSampleRate) / 1000.0f;
  const float kickIntervalSamples = kickIntervalMs * static_cast<float>(kSampleRate) / 1000.0f;
  const float clickDurationSamples = static_cast<float>(kSampleRate) * 0.001f;  // 1ms click

  // Pre-generate kick pattern for the entire test duration
  constexpr int kWarmupBlocks = 50;  // More warmup for transient material
  constexpr int kMeasureBlocks = 100;
  const int totalSamples = (kWarmupBlocks + kMeasureBlocks) * kBlockSize;
  std::vector<float> kickSignal(static_cast<size_t>(totalSamples), 0.0f);

  float phase = 0.0f;
  for (int i = 0; i < totalSamples; ++i) {
    float t = static_cast<float>(i);
    // Find position within current kick interval
    float posInKick = std::fmod(t, kickIntervalSamples);

    if (posInKick < kickDecaySamples * 3.0f) {  // Only generate during active part
      // Amplitude envelope: exponential decay
      float env = kickAmplitude * std::exp(-posInKick / kickDecaySamples);

      // Pitch sweep: exponential from startFreq to endFreq
      float freqT = std::min(posInKick / kickDecaySamples, 1.0f);
      float freq = kickStartFreq * std::pow(kickEndFreq / kickStartFreq, freqT);

      // Phase-continuous oscillator
      if (posInKick < 1.0f) phase = 0.0f;  // Reset phase at kick onset
      phase += 2.0f * kPi * freq / static_cast<float>(kSampleRate);
      if (phase > 2.0f * kPi) phase -= 2.0f * kPi;

      float body = env * std::sin(phase);

      // Add transient click (noise burst, first 1ms)
      float click = 0.0f;
      if (posInKick < clickDurationSamples) {
        float clickEnv = 1.0f - posInKick / clickDurationSamples;
        clickEnv *= clickEnv;  // Squared envelope for sharp attack
        // Simple deterministic "noise" via fast sine harmonics
        click = kickAmplitude * clickEnv * 0.5f *
            (std::sin(phase * 7.0f) + std::sin(phase * 13.0f) + std::sin(phase * 23.0f)) / 3.0f;
      }

      kickSignal[static_cast<size_t>(i)] = body + click;
    }
  }

  // Process through plugin
  processor->releaseResources();
  processor->prepareToPlay(kSampleRate, kBlockSize);

  const int totalChannels = processor->getTotalNumInputChannels() + processor->getTotalNumOutputChannels();
  juce::MidiBuffer midiBuffer;

  std::vector<float> inputAcc, mainAcc, drumsAcc, bassAcc, otherAcc, vocalsAcc;
  size_t reserveSize = static_cast<size_t>(kMeasureBlocks * kBlockSize);
  inputAcc.reserve(reserveSize);
  mainAcc.reserve(reserveSize);
  drumsAcc.reserve(reserveSize);
  bassAcc.reserve(reserveSize);
  otherAcc.reserve(reserveSize);
  vocalsAcc.reserve(reserveSize);

  for (int block = 0; block < kWarmupBlocks + kMeasureBlocks; ++block) {
    juce::AudioBuffer<float> buffer(totalChannels, kBlockSize);
    buffer.clear();

    auto inBus = processor->getBusBuffer(buffer, true, 0);
    int blockOffset = block * kBlockSize;
    for (int i = 0; i < kBlockSize; ++i) {
      float sample = kickSignal[static_cast<size_t>(blockOffset + i)];
      inBus.setSample(0, i, sample);
      inBus.setSample(1, i, sample);
    }

    // Save input before processBlock overwrites shared channels
    std::vector<float> savedInput(static_cast<size_t>(kBlockSize));
    if (block >= kWarmupBlocks) {
      for (int i = 0; i < kBlockSize; ++i)
        savedInput[static_cast<size_t>(i)] = inBus.getSample(0, i);
    }

    processor->processBlock(buffer, midiBuffer);

    // Give inference thread time to process (each 512-sample block = ~11.6ms at 44.1kHz)
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    if (block >= kWarmupBlocks) {
      auto mainBus = processor->getBusBuffer(buffer, false, 0);
      auto drumsBus = processor->getBusBuffer(buffer, false, 1);
      auto bassBus = processor->getBusBuffer(buffer, false, 2);
      auto otherBus = processor->getBusBuffer(buffer, false, 3);
      auto vocalsBus = processor->getBusBuffer(buffer, false, 4);

      for (int i = 0; i < kBlockSize; ++i) {
        inputAcc.push_back(savedInput[static_cast<size_t>(i)]);
        mainAcc.push_back(mainBus.getSample(0, i));
        drumsAcc.push_back(drumsBus.getSample(0, i));
        bassAcc.push_back(bassBus.getSample(0, i));
        otherAcc.push_back(otherBus.getSample(0, i));
        vocalsAcc.push_back(vocalsBus.getSample(0, i));
      }
    }
  }

  // Compute metrics
  auto rmsOf = [](const std::vector<float>& v) {
    float sum = 0.0f;
    for (float s : v) sum += s * s;
    return std::sqrt(sum / static_cast<float>(v.size()));
  };
  auto peakOf = [](const std::vector<float>& v) {
    float p = 0.0f;
    for (float s : v) p = std::max(p, std::abs(s));
    return p;
  };
  auto crestOf = [&](const std::vector<float>& v) {
    float r = rmsOf(v);
    return r > 1e-10f ? peakOf(v) / r : 0.0f;
  };

  // Compute per-kick metrics (find kicks by peak detection)
  // First, compute block-level max error
  float maxSampleError = 0.0f;
  float maxSampleErrorInput = 0.0f;
  for (size_t i = 0; i < mainAcc.size(); ++i) {
    float stemSum = drumsAcc[i] + bassAcc[i] + otherAcc[i] + vocalsAcc[i];
    float err = std::abs(mainAcc[i] - stemSum);
    if (err > maxSampleError) {
      maxSampleError = err;
      maxSampleErrorInput = inputAcc[i];
    }
  }

  // Main bus reads from delayedInputBuffer at ring readPos, so it is delayed
  // by the variable inference pipeline depth (aligned with stems). The delay
  // fluctuates with inference timing jitter, so a fixed-lag comparison isn't
  // meaningful. Find the best lag for diagnostic reporting only.
  float reconErrorRms = 0.0f;
  float reconErrorPeak = 0.0f;
  size_t reconCount = 0;
  size_t lag = 0;
  {
    const size_t maxLag = std::min<size_t>(mainAcc.size() / 2,
                                           static_cast<size_t>(audio_plugin::kOutputChunkSize) * 8);
    float bestRms = std::numeric_limits<float>::max();
    size_t bestLag = 0;
    for (size_t tryLag = 0; tryLag < maxLag; ++tryLag) {
      float rms = 0.0f;
      size_t count = std::min(mainAcc.size(), inputAcc.size()) - tryLag;
      for (size_t i = 0; i < count; ++i) {
        float diff = mainAcc[i + tryLag] - inputAcc[i];
        rms += diff * diff;
      }
      rms = std::sqrt(rms / static_cast<float>(count));
      if (rms < bestRms) {
        bestRms = rms;
        bestLag = tryLag;
      }
    }
    lag = bestLag;
    // Compute final error at best lag
    size_t count = std::min(mainAcc.size(), inputAcc.size()) - lag;
    for (size_t i = 0; i < count; ++i) {
      float diff = mainAcc[i + lag] - inputAcc[i];
      reconErrorRms += diff * diff;
      reconErrorPeak = std::max(reconErrorPeak, std::abs(diff));
      reconCount++;
    }
    reconErrorRms = std::sqrt(reconErrorRms / static_cast<float>(reconCount));
  }

  // Print kick drum report
  std::cerr << "\n";
  std::cerr << "====================================================================\n";
  std::cerr << "  KICK DRUM DIAGNOSTIC REPORT\n";
  std::cerr << "====================================================================\n";
  std::cerr << "  Kick: " << kickStartFreq << "->" << kickEndFreq << " Hz sweep, "
            << kickDecayMs << "ms decay, " << kickAmplitude << " amplitude\n";
  std::cerr << "  Interval: " << kickIntervalMs << "ms (120 BPM)\n";
  std::cerr << "  Warmup: " << kWarmupBlocks << " blocks, Measure: " << kMeasureBlocks << " blocks\n";
  std::cerr << "====================================================================\n\n";

  fprintf(stderr, "%-10s | %-10s %-10s %-10s\n", "Bus", "RMS", "Peak", "Crest");
  fprintf(stderr, "%-10s-+-%-10s-%-10s-%-10s\n", "----------", "----------", "----------", "----------");
  fprintf(stderr, "%-10s | %-10.6f %-10.6f %-10.4f\n", "Input",
          static_cast<double>(rmsOf(inputAcc)), static_cast<double>(peakOf(inputAcc)), static_cast<double>(crestOf(inputAcc)));
  fprintf(stderr, "%-10s | %-10.6f %-10.6f %-10.4f\n", "Main",
          static_cast<double>(rmsOf(mainAcc)), static_cast<double>(peakOf(mainAcc)), static_cast<double>(crestOf(mainAcc)));
  fprintf(stderr, "%-10s | %-10.6f %-10.6f %-10.4f\n", "Drums",
          static_cast<double>(rmsOf(drumsAcc)), static_cast<double>(peakOf(drumsAcc)), static_cast<double>(crestOf(drumsAcc)));
  fprintf(stderr, "%-10s | %-10.6f %-10.6f %-10.4f\n", "Bass",
          static_cast<double>(rmsOf(bassAcc)), static_cast<double>(peakOf(bassAcc)), static_cast<double>(crestOf(bassAcc)));
  fprintf(stderr, "%-10s | %-10.6f %-10.6f %-10.4f\n", "Other",
          static_cast<double>(rmsOf(otherAcc)), static_cast<double>(peakOf(otherAcc)), static_cast<double>(crestOf(otherAcc)));
  fprintf(stderr, "%-10s | %-10.6f %-10.6f %-10.4f\n", "Vocals",
          static_cast<double>(rmsOf(vocalsAcc)), static_cast<double>(peakOf(vocalsAcc)), static_cast<double>(crestOf(vocalsAcc)));

  float inputRms = rmsOf(inputAcc);
  float mainRms = rmsOf(mainAcc);
  float energyRatio = (inputRms > 1e-10f) ? mainRms / inputRms : 0.0f;

  fprintf(stderr, "\n--- Summary ---\n");
  fprintf(stderr, "Energy ratio (main/input): %.4f\n", static_cast<double>(energyRatio));
  fprintf(stderr, "Main-StemSum max error: %.8f (at input=%.6f)\n",
          static_cast<double>(maxSampleError), static_cast<double>(maxSampleErrorInput));
  fprintf(stderr, "Reconstruction error (main vs input, %zu-sample lag): RMS=%.6f  Peak=%.6f\n",
          lag,
          static_cast<double>(reconErrorRms), static_cast<double>(reconErrorPeak));

  // Stem energy distribution
  float drumsRms = rmsOf(drumsAcc);
  float bassRms = rmsOf(bassAcc);
  float otherRms = rmsOf(otherAcc);
  float vocalsRms = rmsOf(vocalsAcc);
  float totalStemRms = drumsRms + bassRms + otherRms + vocalsRms;
  if (totalStemRms > 1e-10f) {
    fprintf(stderr, "\nStem distribution: Drums=%.1f%%  Bass=%.1f%%  Other=%.1f%%  Vocals=%.1f%%\n",
            static_cast<double>(drumsRms / totalStemRms * 100.0f),
            static_cast<double>(bassRms / totalStemRms * 100.0f),
            static_cast<double>(otherRms / totalStemRms * 100.0f),
            static_cast<double>(vocalsRms / totalStemRms * 100.0f));
  }

  // Peak comparison: does the kick transient survive in drums stem?
  float inputPeak = peakOf(inputAcc);
  float drumsPeak = peakOf(drumsAcc);
  float bassPeak = peakOf(bassAcc);
  fprintf(stderr, "\nTransient preservation:\n");
  fprintf(stderr, "  Input peak:  %.6f\n", static_cast<double>(inputPeak));
  fprintf(stderr, "  Drums peak:  %.6f (%.1f%% of input)\n",
          static_cast<double>(drumsPeak), static_cast<double>(drumsPeak / inputPeak * 100.0f));
  fprintf(stderr, "  Bass peak:   %.6f (%.1f%% of input)\n",
          static_cast<double>(bassPeak), static_cast<double>(bassPeak / inputPeak * 100.0f));
  fprintf(stderr, "  Main peak:   %.6f (%.1f%% of input)\n",
          static_cast<double>(peakOf(mainAcc)), static_cast<double>(peakOf(mainAcc) / inputPeak * 100.0f));

  // Crest factor comparison (kick should have high crest factor ~6-10)
  fprintf(stderr, "\nCrest factor comparison (kick drum typical: 6-10):\n");
  fprintf(stderr, "  Input:  %.3f\n", static_cast<double>(crestOf(inputAcc)));
  fprintf(stderr, "  Main:   %.3f\n", static_cast<double>(crestOf(mainAcc)));
  fprintf(stderr, "  Drums:  %.3f\n", static_cast<double>(crestOf(drumsAcc)));
  fprintf(stderr, "  Bass:   %.3f\n", static_cast<double>(crestOf(bassAcc)));

  // Chunk boundary discontinuity analysis
  // Every kOutputChunkSize samples, the model transitions to a new chunk.
  // Discontinuities at boundaries cause ~86Hz artifacts that sound like saturation.
  {
    constexpr int chunkSize = audio_plugin::kOutputChunkSize;
    int latencySamples = processor->getLatencySamples();

    // Compute max sample-to-sample delta at chunk boundaries vs. within chunks
    float maxBoundaryDelta = 0.0f;
    float maxInternalDelta = 0.0f;
    int boundaryCount = 0;
    int internalCount = 0;
    float sumBoundaryDeltaSq = 0.0f;
    float sumInternalDeltaSq = 0.0f;

    for (size_t i = 1; i < drumsAcc.size(); ++i) {
      float delta = std::abs(drumsAcc[i] - drumsAcc[i - 1]);
      // Check if this sample is at a chunk boundary (accounting for latency)
      int sampleInStream = static_cast<int>(i) + latencySamples;
      bool atBoundary = (sampleInStream % chunkSize) == 0;

      if (atBoundary) {
        maxBoundaryDelta = std::max(maxBoundaryDelta, delta);
        sumBoundaryDeltaSq += delta * delta;
        boundaryCount++;
      } else {
        maxInternalDelta = std::max(maxInternalDelta, delta);
        sumInternalDeltaSq += delta * delta;
        internalCount++;
      }
    }

    float rmsBoundary = boundaryCount > 0 ? std::sqrt(sumBoundaryDeltaSq / static_cast<float>(boundaryCount)) : 0.0f;
    float rmsInternal = internalCount > 0 ? std::sqrt(sumInternalDeltaSq / static_cast<float>(internalCount)) : 0.0f;
    float boundaryRatio = (rmsInternal > 1e-10f) ? rmsBoundary / rmsInternal : 0.0f;

    fprintf(stderr, "\nChunk boundary analysis (every %d samples = %.1f Hz):\n",
            chunkSize, static_cast<double>(static_cast<float>(kSampleRate) / static_cast<float>(chunkSize)));
    fprintf(stderr, "  Drums delta at boundaries: max=%.6f  rms=%.6f\n",
            static_cast<double>(maxBoundaryDelta), static_cast<double>(rmsBoundary));
    fprintf(stderr, "  Drums delta within chunks: max=%.6f  rms=%.6f\n",
            static_cast<double>(maxInternalDelta), static_cast<double>(rmsInternal));
    fprintf(stderr, "  Boundary/internal ratio: %.2fx (>2x indicates chunk boundary artifacts)\n",
            static_cast<double>(boundaryRatio));
  }

  std::cerr << "\n====================================================================\n\n";

  // Assertions
  EXPECT_GE(energyRatio, 0.5f) << "Kick drum: energy ratio too low: " << energyRatio;
  EXPECT_LE(energyRatio, 2.0f) << "Kick drum: energy ratio too high: " << energyRatio;
  // Main bus reads from delayedInputBuffer aligned with stems. The pipeline
  // delay varies with inference timing jitter, so the best-lag RMS won't be
  // near-zero. Check that it's reasonable (energy is preserved, not garbled).
  EXPECT_LT(reconErrorRms, 0.5f) << "Kick drum: reconstruction error RMS too high: " << reconErrorRms;
}

TEST_F(BassDiagnosticTest, SampleLevelVerification) {
  if (!isModelLoaded()) {
    GTEST_SKIP() << "Model not loaded, skipping diagnostic test";
  }

  // Check JUCE channel mapping via getBusBuffer
  {
    juce::AudioBuffer<float> probe(12, 1);
    probe.clear();
    auto inBus = processor->getBusBuffer(probe, true, 0);
    auto mainBus = processor->getBusBuffer(probe, false, 0);
    auto drumsBus = processor->getBusBuffer(probe, false, 1);
    auto bassBus = processor->getBusBuffer(probe, false, 2);
    auto otherBus = processor->getBusBuffer(probe, false, 3);
    auto vocalsBus = processor->getBusBuffer(probe, false, 4);

    auto printBusPointers = [](const char* label, const juce::AudioBuffer<float>& bus) {
      if (bus.getNumChannels() <= 0) {
        fprintf(stderr, "%s: disabled (0 ch)\n", label);
        return;
      }
      fprintf(stderr, "%s: ptr %p-%p (%d ch)\n",
              label,
              static_cast<const void*>(bus.getReadPointer(0)),
              static_cast<const void*>(bus.getReadPointer(bus.getNumChannels() - 1)),
              bus.getNumChannels());
    };

    std::cerr << "\n=== JUCE Bus Channel Mapping ===\n";
    printBusPointers("Input  bus 0", inBus);
    printBusPointers("Main   bus 0", mainBus);
    printBusPointers("Drums  bus 1", drumsBus);
    printBusPointers("Bass   bus 2", bassBus);
    printBusPointers("Other  bus 3", otherBus);
    printBusPointers("Vocals bus 4", vocalsBus);
    std::cerr << "\n";
  }

  // Process 100Hz at 0.5 amplitude using getBusBuffer for correct channel access
  constexpr float freq = 100.0f;
  constexpr float amplitude = 0.5f;
  constexpr int kWarmupBlocks = 30;
  constexpr int kMeasureBlocks = 5;
  juce::MidiBuffer midiBuffer;

  processor->releaseResources();
  processor->prepareToPlay(kSampleRate, kBlockSize);

  for (int block = 0; block < kWarmupBlocks + kMeasureBlocks; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    buffer.clear();

    // Write input to the correct input bus channels
    auto inputBus = processor->getBusBuffer(buffer, true, 0);
    for (int i = 0; i < kBlockSize; ++i) {
      int sampleIdx = block * kBlockSize + i;
      float sample = amplitude * std::sin(2.0f * kPi * freq *
          static_cast<float>(sampleIdx) / static_cast<float>(kSampleRate));
      inputBus.setSample(0, i, sample);
      inputBus.setSample(1, i, sample);
    }

    // Save input before processBlock
    std::vector<float> savedInput(kBlockSize);
    if (block == kWarmupBlocks) {
      for (int i = 0; i < kBlockSize; ++i)
        savedInput[static_cast<size_t>(i)] = inputBus.getSample(0, i);
    }

    processor->processBlock(buffer, midiBuffer);

    // Give inference thread time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    if (block == kWarmupBlocks) {
      // Read outputs using getBusBuffer for correct mapping
      auto mainBus = processor->getBusBuffer(buffer, false, 0);
      auto drumsBus = processor->getBusBuffer(buffer, false, 1);
      auto bassBus = processor->getBusBuffer(buffer, false, 2);
      auto otherBus = processor->getBusBuffer(buffer, false, 3);
      auto vocalsBus = processor->getBusBuffer(buffer, false, 4);

      std::cerr << "=== Sample-Level Data via getBusBuffer (block " << block << ", 100Hz) ===\n";
      std::cerr << "i    | SavedInp  | Main      | Drums     | Bass      | Other     | Vocals    | StemSum   | Main-Sum\n";
      std::cerr << "-----+-----------+-----------+-----------+-----------+-----------+-----------+-----------+---------\n";
      for (int i = 0; i < 16; ++i) {
        float inp = savedInput[static_cast<size_t>(i)];
        float main = mainBus.getSample(0, i);
        float dru = drumsBus.getSample(0, i);
        float bas = bassBus.getSample(0, i);
        float oth = otherBus.getSample(0, i);
        float voc = vocalsBus.getSample(0, i);
        float sum = dru + bas + oth + voc;
        float diff = main - sum;
        fprintf(stderr, "%-4d | %+.6f | %+.6f | %+.6f | %+.6f | %+.6f | %+.6f | %+.6f | %+.6f\n",
                i, static_cast<double>(inp), static_cast<double>(main),
                static_cast<double>(dru), static_cast<double>(bas),
                static_cast<double>(oth), static_cast<double>(voc),
                static_cast<double>(sum), static_cast<double>(diff));
      }

      // Compute RMS for the block
      float mainRms = 0.0f, inputRms = 0.0f, stemSumRms = 0.0f;
      float druRms = 0.0f, basRms = 0.0f, othRms = 0.0f, vocRms = 0.0f;
      for (int i = 0; i < kBlockSize; ++i) {
        float inp = savedInput[static_cast<size_t>(i)];
        float main = mainBus.getSample(0, i);
        float dru = drumsBus.getSample(0, i);
        float bas = bassBus.getSample(0, i);
        float oth = otherBus.getSample(0, i);
        float voc = vocalsBus.getSample(0, i);
        inputRms += inp * inp;
        mainRms += main * main;
        druRms += dru * dru;
        basRms += bas * bas;
        othRms += oth * oth;
        vocRms += voc * voc;
        float sum = dru + bas + oth + voc;
        stemSumRms += sum * sum;
      }
      auto rms = [](float s) { return std::sqrt(s / static_cast<float>(kBlockSize)); };
      fprintf(stderr, "\nBlock RMS: input=%.6f  main=%.6f  drums=%.6f  bass=%.6f  other=%.6f  vocals=%.6f  stemSum=%.6f\n",
              static_cast<double>(rms(inputRms)), static_cast<double>(rms(mainRms)),
              static_cast<double>(rms(druRms)), static_cast<double>(rms(basRms)),
              static_cast<double>(rms(othRms)), static_cast<double>(rms(vocRms)),
              static_cast<double>(rms(stemSumRms)));
      fprintf(stderr, "Ratios: main/input=%.4f  stemSum/input=%.4f  main/stemSum=%.4f\n",
              static_cast<double>(rms(mainRms) / rms(inputRms)),
              static_cast<double>(rms(stemSumRms) / rms(inputRms)),
              static_cast<double>(rms(stemSumRms) > 1e-10f ? rms(mainRms) / rms(stemSumRms) : 0.0f));
      std::cerr << "\n";
    }
  }
}

}  // namespace audio_plugin_test
