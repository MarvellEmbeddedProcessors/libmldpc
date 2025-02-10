# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Marvell.

cdata = configuration_data()
cdata.set('PROJECT_NAME', meson.project_name())
cdata.set('LIBMLDPC_VERSION', meson.project_version())
cdata.set('PROJECT_SOURCE_DIR', meson.project_source_root())
cdata.set('DOXYGEN_OUTPUT_DIR', meson.current_build_dir())

doxyfile = configure_file(
  input : 'Doxyfile.in',
  output : 'Doxyfile',
  configuration : cdata,
)

sh = find_program('sh')

custom_target('libmldpc-docs-html',
  input : doxyfile,
  output : 'docs.stamp',
  command : [sh, '-c', 'doxygen "$1" && : > "$2"', 'sh', '@INPUT@', '@OUTPUT@'],
  depend_files : files('MainPage.md', 'InstallPage.md', '../include/mldpc.h'),
  build_by_default : true,
)

meson.add_install_script(
  sh,
  files('install-docs.sh'),
  meson.current_build_dir(),
  get_option('datadir'),
  meson.project_name(),
)
