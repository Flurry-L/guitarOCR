#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p output/logs tools/uv
log_file="output/logs/launcher-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1
echo "GuitarOCR · 环境安装与启动"
echo "日志：$log_file"
uv_binary="$PWD/tools/uv/uv"
if [[ ! -x "$uv_binary" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "请先安装 curl：sudo apt-get install curl"
    exit 1
  fi
  curl --fail --location --retry 3 https://astral.sh/uv/0.12.17/install.sh -o tools/uv/install.sh
  UV_UNMANAGED_INSTALL="$PWD/tools/uv" sh tools/uv/install.sh
fi
export GUITAROCR_UV="$uv_binary"
export PYTHONUTF8=1
"$uv_binary" run --config-file scripts/bootstrap-uv.toml --no-project --python 3.11 scripts/launcher.py "$@"
