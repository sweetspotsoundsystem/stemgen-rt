#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
namespace audio_plugin::parallel_model {
struct Branch {
  std::string_view name, sha256;
  int64_t bytes;
  size_t offset, count;
};
inline constexpr std::array<Branch, 3> kBranches{{
    {"drums",
     "f2a3fc5dd90f45690536370f02d9b9377e6625ab08cb32b3e921371ad134d874",
     12267886, 0, 5},
    {"bass", "1187b414f22344028035caba0a763d2211d13fe8bfce366c5f6b7ee4af77df40",
     8660860, 5, 4},
    {"vocals",
     "f29a3d8fa596ce4672dd4dc77ccf224f2538389940264aad46ab8168311c2148",
     11019332, 9, 4},
}};
}  // namespace audio_plugin::parallel_model
