#!/bin/bash
# Run ONCE on the login node (may need internet for conda): builds revision/libjpegcoef.so against
# libjpeg-turbo. Uses the system jpeglib.h if present, otherwise installs libjpeg-turbo into the conda env.
#   bash revision/build_jpegcoef.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.sh"
activate_env
CC=${CC:-$(command -v gcc || command -v cc || true)}
[ -n "$CC" ] || { echo "no C compiler found (install one with: conda install -y -c conda-forge c-compiler)"; exit 1; }
if [ -f "$CONDA_PREFIX/include/jpeglib.h" ]; then
  INC="$CONDA_PREFIX/include"; LIB="$CONDA_PREFIX/lib"
elif [ -f /usr/include/jpeglib.h ]; then
  INC=/usr/include; LIB=$(dirname "$(ldconfig -p | awk '/libjpeg\.so /{print $NF; exit}')")
else
  conda install -y -c conda-forge libjpeg-turbo
  INC="$CONDA_PREFIX/include"; LIB="$CONDA_PREFIX/lib"
fi
echo "compiler: $CC | jpeglib.h: $INC | libjpeg: $LIB"
"$CC" -O2 -shared -fPIC "$KIT_DIR/jpegcoef.c" -I"$INC" -L"$LIB" -ljpeg -Wl,-rpath,"$LIB" -o "$KIT_DIR/libjpegcoef.so"
python "$KIT_DIR/bitstream.py" --selftest
