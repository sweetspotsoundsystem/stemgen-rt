#include <StemgenRT/PluginProcessor.h>
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <random>

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
  // Crossover should be in sub-bass range
  EXPECT_GT(audio_plugin::kCrossoverFreqHz, 20.0f);
  EXPECT_LT(audio_plugin::kCrossoverFreqHz, 200.0f);
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
};

// --------------------------------------------------------------------------
// Output Sanity Tests (no NaN, Inf, or extreme values)
// --------------------------------------------------------------------------

TEST_F(AudioQualityTest, OutputHasNoNaNWithSineInput) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  generateSineWave(buffer, 440.0f, 0.5f, 0, 2);  // Input channels
  processor->processBlock(buffer, midiBuffer);
  
  // Check all output channels (2-11) for NaN/Inf
  EXPECT_FALSE(hasNaNOrInf(buffer, 2, 10)) << "Output contains NaN or Inf values";
}

TEST_F(AudioQualityTest, OutputHasNoNaNWithNoiseInput) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  generateNoise(buffer, 0.5f, 0, 2);  // Input channels
  processor->processBlock(buffer, midiBuffer);
  
  EXPECT_FALSE(hasNaNOrInf(buffer, 2, 10)) << "Output contains NaN or Inf values with noise input";
}

TEST_F(AudioQualityTest, OutputHasNoNaNWithSilentInput) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  buffer.clear();
  juce::MidiBuffer midiBuffer;
  
  processor->processBlock(buffer, midiBuffer);
  
  EXPECT_FALSE(hasNaNOrInf(buffer, 2, 10)) << "Output contains NaN or Inf with silent input";
}

TEST_F(AudioQualityTest, OutputHasNoNaNWithImpulse) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  generateImpulse(buffer, 100, 1.0f, 0, 2);  // Full-scale impulse
  processor->processBlock(buffer, midiBuffer);
  
  EXPECT_FALSE(hasNaNOrInf(buffer, 2, 10)) << "Output contains NaN or Inf with impulse input";
}

TEST_F(AudioQualityTest, OutputHasNoNaNAfterManyBlocks) {
  juce::MidiBuffer midiBuffer;
  
  // Process many blocks to check for numerical instability over time
  for (int block = 0; block < 1000; ++block) {
    juce::AudioBuffer<float> buffer(12, kBlockSize);
    generateSineWave(buffer, 440.0f, 0.8f, 0, 2);
    processor->processBlock(buffer, midiBuffer);
    
    if (hasNaNOrInf(buffer, 2, 10)) {
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
  float maxOut = getMaxAmplitude(buffer, 2, 10);
  EXPECT_LE(maxOut, 2.0f) << "Output amplitude " << maxOut << " exceeds reasonable bounds";
}

TEST_F(AudioQualityTest, FullScaleInputProducesReasonableOutput) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  // Full-scale input (0 dBFS)
  generateSineWave(buffer, 440.0f, 1.0f, 0, 2);
  processor->processBlock(buffer, midiBuffer);
  
  // Output shouldn't explode even with full-scale input
  float maxOut = getMaxAmplitude(buffer, 2, 10);
  EXPECT_LE(maxOut, 4.0f) << "Full-scale input caused output explosion: " << maxOut;
}

TEST_F(AudioQualityTest, HotInputDoesNotCauseClipping) {
  juce::AudioBuffer<float> buffer(12, kBlockSize);
  juce::MidiBuffer midiBuffer;
  
  // Hot input (+6dB over full scale - simulating DAW clipping scenarios)
  generateSineWave(buffer, 440.0f, 2.0f, 0, 2);
  processor->processBlock(buffer, midiBuffer);
  
  // Should not produce inf/nan even with hot input
  EXPECT_FALSE(hasNaNOrInf(buffer, 2, 10)) << "Hot input caused numerical issues";
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
  float maxOut = getMaxAmplitude(buffer, 2, 10);
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
    
    float currentMax = getMaxAmplitude(buffer, 2, 10);
    
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
    
    EXPECT_FALSE(hasNaNOrInf(buffer, 2, 10)) << "Numerical issues at block " << block;
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
      
      EXPECT_FALSE(hasNaNOrInf(buffer, 2, 10)) << "NaN/Inf after reset cycle " << cycle << " block " << block;
    }
    
    // Reset buffers (simulating transport stop/start)
    processor->resetStreamingBuffers();
  }
}

}  // namespace audio_plugin_test
