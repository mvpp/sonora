#!/usr/bin/env bash
# Build Sonora's Linux .deb inside Docker (no Rust/Bun/-dev packages on the host).
#
# Usage:  bash scripts/build-deb-docker.sh
# Output: src-tauri/target/release/bundle/deb/Sonora_<version>_amd64.deb
# Install: sudo apt install --reinstall ./src-tauri/target/release/bundle/deb/Sonora_*_amd64.deb
#
# Image `sonora-build` = Ubuntu 24.04 + the Tauri deps from
# .github/workflows/build-release.yml + libclang (bindgen for whisper-rs; the
# GitHub runner ships it preinstalled) + Rust stable + Bun. The cargo registry
# lives in the `sonora-cargo` volume and `target/` stays in the checkout, so
# rebuilds are incremental. Outputs are chowned back to the calling user.
set -euo pipefail

cd "$(dirname "$0")/.."

docker build -t sonora-build - <<'EOF'
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo \
    BUN_INSTALL=/usr/local/bun
ENV PATH=/usr/local/cargo/bin:/usr/local/bun/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
      libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev \
      libdbus-1-dev libasound2-dev libssl-dev patchelf cmake build-essential \
      libclang-dev \
      pkg-config file curl ca-certificates unzip git \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
RUN curl -fsSL https://bun.sh/install | bash
EOF

docker run --rm \
  -v "$PWD":/src -w /src \
  -v sonora-cargo:/usr/local/cargo/registry \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  sonora-build bash -euc '
    trap "chown -R \$HOST_UID:\$HOST_GID node_modules src-tauri/target dist 2>/dev/null || true" EXIT
    bun install --frozen-lockfile
    bun run tauri build --bundles deb
  '

ls -1 src-tauri/target/release/bundle/deb/*.deb
