#pragma once

#include "PluginProcessor.h"
#include "BuildConfig.h"

namespace audio_plugin {

#if !STEMGENRT_DEBUG_UI
// Release build - just display logo
class AudioPluginAudioProcessorEditor : public juce::AudioProcessorEditor {
public:
  explicit AudioPluginAudioProcessorEditor(AudioPluginAudioProcessor&);
  ~AudioPluginAudioProcessorEditor() override;

  void paint(juce::Graphics&) override;
  void resized() override;

private:
  [[maybe_unused]] AudioPluginAudioProcessor& processorRef;

#if HAS_LOGO_ASSET
  juce::Image logoImage;
#endif

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioPluginAudioProcessorEditor)
};

#else
// Debug build - display status info
class AudioPluginAudioProcessorEditor : public juce::AudioProcessorEditor,
                                        private juce::Timer {
public:
  explicit AudioPluginAudioProcessorEditor(AudioPluginAudioProcessor&);
  ~AudioPluginAudioProcessorEditor() override;

  void paint(juce::Graphics&) override;
  void resized() override;

private:
  void timerCallback() override;

  AudioPluginAudioProcessor& processorRef;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioPluginAudioProcessorEditor)
};
#endif

}  // namespace audio_plugin
