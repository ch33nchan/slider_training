#!/bin/bash
# Apply Klein compatibility patches to ai-toolkit
# Run from the SliderTraining-Klein root directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_TOOLKIT_DIR="$(dirname "$SCRIPT_DIR")/ai-toolkit"

if [ ! -d "$AI_TOOLKIT_DIR" ]; then
    echo "Error: ai-toolkit not found at $AI_TOOLKIT_DIR"
    exit 1
fi

cd "$AI_TOOLKIT_DIR"
git apply "$SCRIPT_DIR/0001-klein-ai-toolkit-fixes.patch"
echo "Patches applied successfully"
