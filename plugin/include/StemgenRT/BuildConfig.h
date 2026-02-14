#pragma once

// Build-time feature flags.
//
// We intentionally avoid keying "debug UI" off NDEBUG directly because the
// build system (JUCE recommended flags + missing CMAKE_BUILD_TYPE) may define
// NDEBUG even when doing a local dev build. This flag is controlled from CMake.

#ifndef STEMGENRT_DEBUG_UI
#define STEMGENRT_DEBUG_UI 0
#endif

