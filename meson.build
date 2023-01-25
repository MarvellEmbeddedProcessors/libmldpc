# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Marvell.

project('libtvm_dpdk_mldev', 'c',
  version : '0.1')

pkgconfig = find_program('pkg-config')
cflags_cmd = run_command(pkgconfig, '--cflags', 'libdpdk', check: true)
ldflags_cmd = run_command(pkgconfig, '--static', '--libs', 'libdpdk', check: true)

deps = dependency('jansson')

cflags = cflags_cmd.stdout().split()
ldflags = ldflags_cmd.stdout().split()

includes = include_directories('include', )

subdir('src')
subdir('include')
subdir('test')
