#!/usr/bin/env bash
# Install Node deps + macOS OpenSSL linkage for the QVAC SDK sidecar.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/sidecar"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js >= 22.17 is required. Install from https://nodejs.org/"
  exit 1
fi

NODE_MAJOR="$(node -p "process.versions.node.split('.')[0]")"
if [[ "$NODE_MAJOR" -lt 22 ]]; then
  echo "Node.js >= 22.17 required (found $(node -v))"
  exit 1
fi

npm install

# --- macOS: QVAC .bare workers hardcode Homebrew OpenSSL 3 paths ---
if [[ "$(uname -s)" == "Darwin" ]]; then
  OPENSSL_PREFIX="${QVAC_OPENSSL_PREFIX:-$HOME/.local/qvac-openssl}"
  if [[ ! -f "$OPENSSL_PREFIX/lib/libssl.3.dylib" ]]; then
    if command -v brew >/dev/null 2>&1; then
      echo "==> Installing openssl@3 via Homebrew..."
      brew install openssl@3
      OPENSSL_PREFIX="$(brew --prefix openssl@3)"
    else
      echo "==> Building OpenSSL 3 into $OPENSSL_PREFIX (no Homebrew)..."
      SRC="/tmp/qvac-openssl-src-$$"
      mkdir -p "$SRC" "$OPENSSL_PREFIX"
      curl -fsSL -o "$SRC/openssl.tar.gz" \
        "https://github.com/openssl/openssl/releases/download/openssl-3.3.2/openssl-3.3.2.tar.gz"
      tar -xzf "$SRC/openssl.tar.gz" -C "$SRC"
      (
        cd "$SRC"/openssl-3.3.2
        ./Configure darwin64-arm64-cc --prefix="$OPENSSL_PREFIX" --openssldir="$OPENSSL_PREFIX/ssl" shared no-docs
        make -j"$(sysctl -n hw.ncpu)"
        make install_sw
      )
      rm -rf "$SRC"
    fi
  fi

  echo "==> Linking QVAC native addons to $OPENSSL_PREFIX/lib ..."
  while IFS= read -r bare; do
    if otool -L "$bare" 2>/dev/null | grep -q '/opt/homebrew/opt/openssl@3\|qvac-openssl'; then
      install_name_tool -change \
        /opt/homebrew/opt/openssl@3/lib/libssl.3.dylib \
        "$OPENSSL_PREFIX/lib/libssl.3.dylib" "$bare" 2>/dev/null || true
      install_name_tool -change \
        /opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib \
        "$OPENSSL_PREFIX/lib/libcrypto.3.dylib" "$bare" 2>/dev/null || true
      codesign --force --sign - "$bare" >/dev/null 2>&1 || true
    fi
  done < <(find "$ROOT/sidecar/node_modules/@qvac" -name '*.bare' 2>/dev/null)

  # Space-free model symlink (SDK worker breaks on %20 paths)
  MODEL_SRC="$ROOT/models/medpsy-4b-q4_k_m-imat.gguf"
  if [[ -f "$MODEL_SRC" ]]; then
    mkdir -p "$HOME/.local/qvac-models"
    ln -sfn "$MODEL_SRC" "$HOME/.local/qvac-models/medpsy-4b-q4_k_m-imat.gguf"
    echo "==> Model link: $HOME/.local/qvac-models/medpsy-4b-q4_k_m-imat.gguf"
  fi
fi

echo ""
echo "Sidecar ready. Start with:"
echo "  cd sidecar && npm start"
echo ""
echo "Optional: export QVAC_MODEL_PATH=/path/to/medpsy-*.gguf"
echo "Health check: curl http://127.0.0.1:8787/health"
