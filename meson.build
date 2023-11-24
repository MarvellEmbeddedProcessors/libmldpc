# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Marvell.

project('libmldpc', 'c',
  version : '1.0')

ext_deps = []

pkgconfig = find_program('pkg-config')
cflags = run_command(pkgconfig, '--cflags', 'libdpdk', check: true).stdout().split()
ldflags = run_command(pkgconfig, '--static', '--libs', 'libdpdk', check: true).stdout().split()

jansson_dep = dependency('jansson', method: 'pkg-config', required: true)
if jansson_dep.found()
  ext_deps += jansson_dep
endif

includes = include_directories('include', )

subdir('include')
subdir('src')
subdir('test')

install_data('config.json', install_dir: 'share/mldpc', install_mode: 'rw-r--r--')
