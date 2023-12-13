# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Marvell.

cdata = configuration_data()
cdata.set('PROJECT_SOURCE_DIR', meson.project_source_root())
cdata.set('LIBMLDPC_VERSION', meson.project_version())
cdata.set('PROJECT_NAME', meson.project_name())

doxyfile = configure_file(input: 'Doxyfile.in',
                          output: 'Doxyfile',
                          configuration: cdata,
                          install: false)

doc_target = run_target('doc',command: [doxygen, doxyfile])
