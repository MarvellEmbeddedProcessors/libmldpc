# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Marvell.

project('libmldpc', 'c',
  version : '1.0')

ext_deps = []

libdpdk_dep = dependency('libdpdk', method: 'pkg-config', required: true)
if libdpdk_dep.found()
  ext_deps += libdpdk_dep
endif

jansson_dep = dependency('jansson', method: 'pkg-config', required: true)
if jansson_dep.found()
  ext_deps += jansson_dep
endif

includes = include_directories('include', )

subdir('include')
subdir('src')
subdir('test')

install_data('config.json', install_dir: 'share/mldpc', install_mode: 'rw-r--r--')
