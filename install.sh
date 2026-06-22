#!/usr/bin/env bash
set -e

export GIT_TERMINAL_PROMPT=0        # Git: never prompt, fail cleanly if auth needed
export GIT_ASKPASS=/usr/bin/echo    # Git: silent credential return instead of prompt
export PIP_NO_INPUT=1               # Pip: non-interactive, no prompts

echo "==> Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10+." >&2
    exit 1
fi

echo "==> Checking pip..."
PYTHON=$(command -v python3)
if ! "$PYTHON" -m pip --version &>/dev/null; then
    echo "ERROR: pip not available for $PYTHON. Install pip first." >&2
    exit 1
fi

echo "==> Installing Python dependencies..."
"$PYTHON" -m pip install pyyaml

echo "==> Checking git..."
if ! command -v git &>/dev/null; then
    echo "ERROR: git not found. Please install git." >&2
    exit 1
fi

TMP_DIR="/tmp/llm-fusion-install"
SKILL_DIR="${HERMES_HOME:-$HOME/.hermes}/skills"

echo "==> Cleaning temp directory..."
rm -rf "$TMP_DIR"

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
echo ""
echo "    Next steps:"
echo "    1. Set your API key:"
echo "         export OPENCODE_GO_API_KEY=YOUR_KEY_HERE"
echo "       Or add it to ~/.hermes/.env:"
echo "         echo 'OPENCODE_GO_API_KEY=YOUR_KEY_HERE' >> ~/.hermes/.env"
echo ""
echo "    2. Restart Hermes or run: hermes skills reload (v16.x) or hermes skills reset (v17.0+)"
echo "       (\"hermes skills reset\" replaced \"hermes skills reload\" in Hermes v17.0)"
echo ""
echo "    3. Try it:"
echo "         llm-fusion Explain recursion in one sentence"
echo ""
