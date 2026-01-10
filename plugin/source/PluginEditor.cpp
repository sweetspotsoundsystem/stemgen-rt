#include "StemgenRT/PluginEditor.h"
#include "StemgenRT/PluginProcessor.h"

#if HAS_LOGO_ASSET
#include "BinaryData.h"
#endif

namespace audio_plugin {

#ifdef NDEBUG
// =============================================================================
// Release build - just display logo
// =============================================================================

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor(
    AudioPluginAudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p) {
  setSize(300, 300);

#if HAS_LOGO_ASSET
  logoImage = juce::ImageCache::getFromMemory(BinaryData::logo_png,
                                              BinaryData::logo_pngSize);
#endif
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor() = default;

void AudioPluginAudioProcessorEditor::paint(juce::Graphics& g) {
  g.fillAll(juce::Colours::black);

#if HAS_LOGO_ASSET
  if (logoImage.isValid()) {
    auto bounds = getLocalBounds().reduced(20).toFloat();
    float scale = juce::jmin(bounds.getWidth() / logoImage.getWidth(),
                             bounds.getHeight() / logoImage.getHeight());
    g.drawImage(logoImage,
                bounds.withSizeKeepingCentre(logoImage.getWidth() * scale,
                                             logoImage.getHeight() * scale));
  }
#else
  g.setColour(juce::Colours::white);
  g.setFont(24.0f);
  g.drawFittedText("StemgenRT", getLocalBounds(), juce::Justification::centred,
                   1);
#endif
}

void AudioPluginAudioProcessorEditor::resized() {}

#else
// =============================================================================
// Debug build - display status info
// =============================================================================

AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor(
    AudioPluginAudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p) {
  setSize(300, 300);
  startTimer(100);
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor() {
  stopTimer();
}

void AudioPluginAudioProcessorEditor::paint(juce::Graphics& g) {
  g.fillAll(
      getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));

  g.setColour(juce::Colours::white);
  g.setFont(15.0f);
  juce::Rectangle<int> area = getLocalBounds().reduced(10);

  // Title
  g.drawFittedText("StemgenRT", area.removeFromTop(30),
                   juce::Justification::centred, 1);

  area.removeFromTop(10);  // Spacing

  // Status
  const auto status = processorRef.getOrtStatusString();
  g.drawFittedText(status, area.removeFromTop(24),
                   juce::Justification::centred, 1);

  area.removeFromTop(20);  // Spacing

  // Latency display
  int latencySamples = processorRef.getLatencySamples();
  double latencyMs = processorRef.getLatencyMs();

  juce::String latencyText = juce::String::formatted(
      "Latency: %.1f ms (%d samples)", latencyMs, latencySamples);
  g.drawFittedText(latencyText, area.removeFromTop(24),
                   juce::Justification::centred, 1);
}

void AudioPluginAudioProcessorEditor::resized() {}

void AudioPluginAudioProcessorEditor::timerCallback() { repaint(); }

#endif

}  // namespace audio_plugin
