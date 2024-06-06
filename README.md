<!--- # SPDX-License-Identifier: BSD-3-Clause --->
<!--- # Copyright (c) 2024 Marvell.           --->

# MLDPC

**Machine Learning DataPlane Callbacks Library**


**MLDPC** is a wrapper library for DPDK rte_mldev APIs. This wrapper simplifies
the call sequence used to run an inference operation through DPDK APIs.

#### Dependencies:

| Package | Version  |
|---------|----------|
| DPDK    | >= 23.11 |
| Jansson | >= 2.13  |


#### Build Steps:

```sh
meson setup build --cross-file config/arm64_cn10k_linux_gcc --prefix <install_dir>
ninja -C build
ninja -C build install
 ```
