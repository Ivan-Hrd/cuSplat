# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set(CUSPLAT_RAPIDS_VERSION "26.08")
set(CUSPLAT_RAPIDS_BRANCH "release/${CUSPLAT_RAPIDS_VERSION}")

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/CUSPLAT_RAPIDS.cmake)
  file(DOWNLOAD
          https://raw.githubusercontent.com/rapidsai/rapids-cmake/${CUSPLAT_RAPIDS_BRANCH}/RAPIDS.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/CUSPLAT_RAPIDS.cmake
          STATUS _dl_status
  )
  list(GET _dl_status 0 _dl_code)
  if(NOT _dl_code EQUAL 0)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/CUSPLAT_RAPIDS.cmake)
    message(FATAL_ERROR "Failed to download RAPIDS.cmake: ${_dl_status}")
  endif()
endif()

set(rapids-cmake-version "${CUSPLAT_RAPIDS_VERSION}")
set(rapids-cmake-branch "${CUSPLAT_RAPIDS_BRANCH}")
include(${CMAKE_CURRENT_BINARY_DIR}/CUSPLAT_RAPIDS.cmake)