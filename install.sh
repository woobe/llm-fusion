#!/usr/bin/env bash
set -e

TMP_DIR="/tmp/llm-fusion-install"
SKILL_DIR="${HERMES_HOME:-$HOME/.hermes}/skills"

echo "==> Cloning llm-fusion repo..."
git clone https://github.com/woobe/llm-fusion.git "$TMP_DIR"

echo "==> Checking Hermes skills directory..."
if [ ! -d "$SKILL_DIR" ]; then
    echo "ERROR: $SKILL_DIR does not exist. Make sure Hermes is installed." >&2
    rm -rf "$TMP_DIR"
    exit 1
fi

echo "==> Installing llm-fusion skill..."
cp -r "$TMP_DIR/skills/llm-fusion" "$SKILL_DIR/"

echo "==> Cleaning up..."
rm -rf "$TMP_DIR"

echo "==> Success! llm-fusion skill installed to $SKILL_DIR/llm-fusion"
