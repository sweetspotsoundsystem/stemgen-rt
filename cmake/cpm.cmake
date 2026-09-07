set(CPM_DOWNLOAD_VERSION 0.40.8)
set(CPM_DOWNLOAD_SHA256
    78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791)

set(CPM_DOWNLOAD_LOCATION "${LIB_DIR}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

# Expand relative path. This is important if the provided path contains a tilde (~)
get_filename_component(CPM_DOWNLOAD_LOCATION ${CPM_DOWNLOAD_LOCATION} ABSOLUTE)

function(download_cpm)
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
       EXPECTED_HASH SHA256=${CPM_DOWNLOAD_SHA256}
       TLS_VERIFY ON
  )
endfunction()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  download_cpm()
else()
  # resume download if it previously failed
  file(READ ${CPM_DOWNLOAD_LOCATION} check)
  if("${check}" STREQUAL "")
    download_cpm()
  endif()
  unset(check)
endif()

file(SHA256 "${CPM_DOWNLOAD_LOCATION}" CPM_DOWNLOADED_SHA256)
if(NOT "${CPM_DOWNLOADED_SHA256}" STREQUAL "${CPM_DOWNLOAD_SHA256}")
  message(FATAL_ERROR
    "CPM.cmake SHA-256 mismatch: ${CPM_DOWNLOADED_SHA256}")
endif()

include(${CPM_DOWNLOAD_LOCATION})
