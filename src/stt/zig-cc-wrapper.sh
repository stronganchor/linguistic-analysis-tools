#!/usr/bin/env bash
exec $LINUX_STT_ENV/.venv-vibevoice/lib/python3.12/site-packages/ziglang/zig cc \
  -I$LINUX_STT_ENV/python312-dev-headers/usr/include \
  -I$LINUX_STT_ENV/python312-dev-headers/usr/include/x86_64-linux-gnu/python3.12 \
  -I$LINUX_STT_ENV/python312-dev-headers/usr/include/python3.12 \
  "$@"
