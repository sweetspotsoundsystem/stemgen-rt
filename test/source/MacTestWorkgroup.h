#pragma once

#if defined(__APPLE__)
#include <AudioToolbox/AudioToolbox.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_basics/native/juce_AudioWorkgroup_mac.h>
#include <os/object.h>
#include <os/workgroup.h>

namespace audio_plugin_test {

// Test-created interval groups exercise native attachment and ownership.
// They do not stand in for physical DAW-host workgroup qualification.
class MacTestWorkgroup {
 public:
  MacTestWorkgroup()
      : handle_(AudioWorkIntervalCreate("StemgenRT lifecycle test",
                                        OS_CLOCK_MACH_ABSOLUTE_TIME,
                                        nullptr)) {}
  ~MacTestWorkgroup() {
    if (handle_ != nullptr) {
      os_release(handle_);
    }
  }
  MacTestWorkgroup(const MacTestWorkgroup&) = delete;
  MacTestWorkgroup& operator=(const MacTestWorkgroup&) = delete;
  os_workgroup_t get() const { return handle_; }
  juce::AudioWorkgroup context() const {
    return juce::makeRealAudioWorkgroup(handle_);
  }

 private:
  os_workgroup_t handle_;
};

}  // namespace audio_plugin_test
#endif
