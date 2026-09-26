#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p output/logs tools/uv
log_file="output/logs/launcher-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1
echo "GuitarOCR 环境安装与启动"
echo "日志：$log_file"
uv_binary="$PWD/tools/uv/uv"
if [[ ! -x "$uv_binary" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "请先安装 curl：sudo apt-get install curl"
    exit 1
  fi
  curl --fail --location --retry 3 --connect-timeout 15 --max-time 120 https://astral.sh/uv/0.12.17/install.sh -o tools/uv/install.sh
  UV_UNMANAGED_INSTALL="$PWD/tools/uv" sh tools/uv/install.sh
fi
export GUITAROCR_UV="$uv_binary"
export PYTHONUTF8=1
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-30}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-2}"
if ! bootstrap_python=$("$uv_binary" --config-file scripts/bootstrap-uv.toml python find --no-python-downloads 3.11 2>/dev/null); then
  python_mirror="${UV_PYTHON_INSTALL_MIRROR:-https://mirrors.nju.edu.cn/github-release/astral-sh/python-build-standalone}"
  echo "正在下载 Python，默认使用南京大学镜像。"
  if ! "$uv_binary" --config-file scripts/bootstrap-uv.toml python install 3.11 --no-bin --no-registry --mirror "$python_mirror"; then
    echo "Python 镜像下载失败，正在使用官方源重试。"
    "$uv_binary" --config-file scripts/bootstrap-uv.toml python install 3.11 --no-bin --no-registry --mirror https://github.com/astral-sh/python-build-standalone/releases/download
  fi
  bootstrap_python=$("$uv_binary" --config-file scripts/bootstrap-uv.toml python find --no-python-downloads 3.11)
fi
"$uv_binary" run --config-file scripts/bootstrap-uv.toml --no-project --no-python-downloads --python "$bootstrap_python" scripts/launcher.py "$@"
