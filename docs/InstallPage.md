<!--
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Marvell.
-->

# Steps to generate the libmdpc library

## Install the dependencies

### 1. Jansson

```
wget -N http://digip.org/jansson/releases/jansson-2.13.tar.gz
tar -xzvf jansson-2.13.tar.gz
cd jansson-2.13
cmake \
    -B ./build \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DJANSSON_BUILD_SHARED_LIBS=ON
    .
make -C build
make -C build install
```

### 2. DataPlane Development Kit

```
For generating libmldpc, we need to pre-build and install the DPDK library.
Steps for installing DPDK library is not listed here.
Here, is the reference to build DataPlane library:
https://doc.dpdk.org/guides/linux_gsg/build_dpdk.html
```

## Set the environment variable

Add path to DPDK and Jansson pkg-config (.pc) files to PKG\_CONFIG\_PATH
```
Example:
export PKG_CONFIG_PATH=${INSTALL_DIR}/lib/pkgconfig
```

## Steps to build and install libmldpc

```
meson setup -Denable_docs=true build
ninja -C build doc
```
