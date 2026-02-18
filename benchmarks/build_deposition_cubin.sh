#!/usr/bin/env bash
set -euo pipefail

# Build precompiled CUBIN for experimental Nm=3 raw deposition kernels.
#
# Usage:
#   benchmarks/build_deposition_cubin.sh [sm_arch] [output_path]
#
# Example:
#   benchmarks/build_deposition_cubin.sh sm_80 \
#     fbpic/particles/deposition/kernels/deposition_nm3_raw_sm80.cubin

SM_ARCH="${1:-sm_80}"
OUT_PATH="${2:-fbpic/particles/deposition/kernels/deposition_nm3_raw_${SM_ARCH}.cubin}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_PATH="${REPO_ROOT}/fbpic/particles/deposition/kernels/deposition_nm3_raw.cu"
OUT_ABS="${REPO_ROOT}/${OUT_PATH}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Error: nvcc not found in PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_ABS}")"

echo "Building CUBIN:"
echo "  source : ${SRC_PATH}"
echo "  arch   : ${SM_ARCH}"
echo "  output : ${OUT_ABS}"

nvcc -O3 -std=c++11 -arch="${SM_ARCH}" -cubin "${SRC_PATH}" -o "${OUT_ABS}"

echo
echo "Done. To use this backend:"
echo "  export FBPIC_DEPOSITION_BACKEND=cubin"
echo "  export FBPIC_DEPOSITION_CUBIN_PATH=${OUT_ABS}"
