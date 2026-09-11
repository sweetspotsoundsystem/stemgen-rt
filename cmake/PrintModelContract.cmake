# Emit machine-readable values for shell qualification tools. Keep model
# identity and timing authoritative in QualifiedModelContract.cmake.
include("${CMAKE_CURRENT_LIST_DIR}/QualifiedModelContract.cmake")
message("model_sha256=${STEMGENRT_QUALIFIED_MODEL_SHA256}")
message("model_bytes=${STEMGENRT_QUALIFIED_MODEL_BYTE_SIZE}")
message("sample_rate=${STEMGENRT_QUALIFIED_SAMPLE_RATE}")
message("callback_samples=${STEMGENRT_QUALIFIED_HOP_SAMPLES}")
message("pdc_samples=${STEMGENRT_QUALIFIED_PLUGIN_LATENCY_SAMPLES}")
