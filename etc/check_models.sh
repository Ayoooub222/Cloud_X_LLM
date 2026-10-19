#!/usr/bin/env bash
cd "$(dirname "$0")/.."
set -a; source .env; set +a
for f in "$IMAGE_DIFFUSION" "$IMAGE_VAE" "$IMAGE_LLM" "$AUDIO_MODEL_DIR/config.json"; do
  [ -f "$MODELS_DIR/$f" ] && echo "OK        $f" || echo "MANQUANT  $f"
done
