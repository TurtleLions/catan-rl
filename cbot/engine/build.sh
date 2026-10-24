#!/bin/bash
# Builds cbot/_engine*.so (the native engine and search) with the venv's Python.
set -e
cd "$(dirname "$0")"
PY="$(cd ../.. && pwd)/.venv/bin/python"
OUT=../_engine$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O3 -march=native -std=c++17 -shared -fPIC -Wall -Wno-sign-compare \
  $($PY -m pybind11 --includes) module.cpp -o "$OUT" -lpthread
echo "built $OUT"
