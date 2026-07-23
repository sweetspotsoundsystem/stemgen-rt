#pragma once

#include <array>
#include <vector>
#include "Constants.h"

namespace audio_plugin {

// Keeps quiet fullband input near the model's nominal operating level while
// preserving coherent amplitude-domain state across streaming hops. This runs
// only on the serialized inference worker; allocate() and reset() are control-
// path operations, while prepare() and commit() allocate no memory.
class ModelInputNormalizer {
public:
  void allocate();
  void reset();

  // Calculate one stereo-linked boost from the exact raw analysis window,
  // migrate normalizedPast into that gain domain, fill the graph's normalized
  // current hop, and return the exact raw current hop for Main/residual
  // alignment.
  bool prepare(const std::array<std::vector<float>, kNumChannels>& inputChunk,
               std::vector<float>& normalizedCurrent,
               std::vector<float>& normalizedPast,
               std::array<std::vector<float>, kNumChannels>& alignedInput,
               float& normalizationGain);

  // Advance raw alignment/gain state only after a successful graph run.
  void commit(const std::array<std::vector<float>, kNumChannels>& inputChunk,
              float normalizationGain);

  bool hasPastAudio() const { return hasPastAudio_; }

private:
  std::vector<float> rawPastAudio_;
  float modelInputGain_{1.0f};
  bool hasPastAudio_{false};
};

}  // namespace audio_plugin
