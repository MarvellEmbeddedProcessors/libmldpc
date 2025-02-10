# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Marvell.

test_sources = files('main.c')

# The test app directly includes DPDK headers, so dpdk_dep is listed explicitly.
# mldpc_dep brings in libmldpc (shared library) and its include path.
executable('test-mldpc',
  sources : test_sources,
  include_directories : inc_dir,
  dependencies : [dpdk_dep, mldpc_dep],
  install : false,
)
