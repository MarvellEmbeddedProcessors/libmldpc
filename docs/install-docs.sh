#!/bin/sh

set -eu

build_dir=$1
datadir=$2
project_name=$3
src_dir=$build_dir/html
prefix=${MESON_INSTALL_DESTDIR_PREFIX:-${MESON_INSTALL_PREFIX:-}}

if [ ! -d "$src_dir" ]; then
    echo "Documentation output directory not found: $src_dir" >&2
    exit 1
fi

case "$datadir" in
    /*)
        install_root="$prefix$datadir"
        ;;
    *)
        install_root="$prefix/$datadir"
        ;;
esac

dest_dir="$install_root/doc/$project_name"
mkdir -p "$dest_dir"
cp -R "$src_dir"/. "$dest_dir"/
