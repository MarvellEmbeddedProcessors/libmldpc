# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Marvell.

mldpc_sources = files('mldpc.c')

mldpc_lib = shared_library('mldpc',
  sources : mldpc_sources,
  include_directories : inc_dir,
  dependencies : [jansson_dep, dpdk_dep],
  # ALLOW_EXPERIMENTAL_API suppresses the __rte_experimental deprecation warning.
  c_args : ['-DALLOW_EXPERIMENTAL_API'],
  version : meson.project_version(),
  install : true,
)

# Expose as a dependency for consumers within this build (e.g. test/)
mldpc_dep = declare_dependency(
  link_with : mldpc_lib,
  include_directories : inc_dir,
)
