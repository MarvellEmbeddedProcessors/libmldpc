# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Marvell.

project('libmldpc', 'c',
  version : '1.0',
  default_options : ['c_std=c11'],
)

# _GNU_SOURCE is required to expose POSIX/GNU extensions (e.g. strnlen) that
# DPDK headers rely on but that are not visible under strict -std=c11.
add_project_arguments('-D_GNU_SOURCE', language : 'c')

# Dependencies
# jansson: JSON parsing used in src/mldpc.c to load the EAL config file
jansson_dep = dependency('jansson', required : true)

# libdpdk: DPDK EAL, mempool, memzone, and ML device APIs
dpdk_dep = dependency('libdpdk', required : true)

# Shared include directory
inc_dir = include_directories('include')

subdir('include')
subdir('src')

if get_option('enable_tests')
  subdir('test')
endif

if get_option('enable_docs')
  doxygen = find_program('doxygen', required : false)
  if doxygen.found()
    subdir('docs')
  else
    warning('enable_docs was set but doxygen was not found; skipping documentation build')
  endif
endif
