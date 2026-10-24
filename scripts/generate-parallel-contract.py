import json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
p=HERE;m=json.loads((p/'model/parallel/manifest.json').read_text())
lines=['# Generated from model/parallel/manifest.json; experimental partitions.']
for b in m['branches']:
 name=b['source'].upper()
 lines += [f'set(STEMGENRT_PARALLEL_{name}_SHA256 "{b["sha256"]}")',f'set(STEMGENRT_PARALLEL_{name}_BYTES {b["bytes"]})']
(p/'cmake/ParallelModelContract.cmake').write_text('\n'.join(lines)+'\n')
header='''#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
namespace audio_plugin::parallel_model {
struct Branch { std::string_view name, sha256; int64_t bytes; size_t offset, count; };
inline constexpr std::array<Branch, 3> kBranches{{
'''
for b in m['branches']:
 header+=f'  {{"{b["source"]}", "{b["sha256"]}", {b["bytes"]}, {b["state_offset"]}, {b["state_count"]}}},\n'
header+='}};\n}  // namespace audio_plugin::parallel_model\n'
(p/'plugin/include/StemgenRT/ParallelModelContract.h').write_text(header)
