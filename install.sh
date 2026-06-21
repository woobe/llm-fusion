#!/usr/bin/env bash
set -e

TMP_DIR="/tmp/llm-fusion-install"

echo "==> Cloning llm-fusion repo..."
git clone https://github.com/woobe/llm-fusion.git "$TMP_DIR"

echo "==> Checking ~/.hermes/skills/ directory..."
if [ ! -d "$HOME/.hermes/skills" ]; then
    echo "ERROR: ~/.hermes/skills/ does not exist. Create it first with: mkdir -p ~/.hermes/skills" >&2
    rm -rf "$TMP_DIR"
    exit 1
fi

echo "==> Installing llm-fusion skill..."
cp -r "$TMP_DIR/skills/llm-fusion" "$HOME/.hermes/skills/"

echo "==> Cleaning up..."
rm -rf "$TMP_DIR"

echo "==> Success! llm-fusion skill installed to ~/.hermes/skills/llm-fusion"
