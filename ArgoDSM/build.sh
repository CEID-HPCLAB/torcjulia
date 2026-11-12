#!/bin/bash
set -e

SRC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ARGODSM_DIR="$SRC/argodsm"
BUILD_DIR="$ARGODSM_DIR/build"
ARTIFACT_DIR="$SRC/../torcjulia/src/argodsm/lib"

export JULIA_PROJECT="${JULIA_PROJECT:-$(cd "$SRC/../torcjulia" && pwd)}"

mkdir -p "$ARGODSM_DIR" "$BUILD_DIR" "$ARTIFACT_DIR"
cd "$ARGODSM_DIR"

if [ ! -d ".git" ]; then
    git init
    git remote add origin https://github.com/etascale/argodsm.git
    git fetch origin
    git checkout -t origin/master
    git submodule update --init --recursive
else
    git pull
    git submodule update --init --recursive
fi

cd "$BUILD_DIR"
unset LD_PRELOAD

if ! julia -e 'using MPI; nothing' 2>/dev/null; then
    echo "Error: Julia MPI not available" >&2
    exit 1
fi

JULIA_MPI_LIB=$(julia -e 'using MPI; print(MPI.libmpi)')
JULIA_MPI_LIB_DIR=$(dirname "$JULIA_MPI_LIB")
JULIA_MPI_INCLUDE=$(julia -e 'using MPI; print(joinpath(dirname(dirname(MPI.libmpi)), "include"))')

cmake "$ARGODSM_DIR" \
    -DARGO_BACKEND_MPI=ON \
    -DARGO_BACKEND_SINGLENODE=ON \
    -DARGO_TESTS=OFF \
    -DBUILD_DOCUMENTATION=OFF \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_FLAGS="-I$JULIA_MPI_INCLUDE" \
    -DCMAKE_CXX_FLAGS="-I$JULIA_MPI_INCLUDE" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L$JULIA_MPI_LIB_DIR -Wl,-rpath,$JULIA_MPI_LIB_DIR" \
    -DCMAKE_EXE_LINKER_FLAGS="-L$JULIA_MPI_LIB_DIR -Wl,-rpath,$JULIA_MPI_LIB_DIR" \
    -DCMAKE_INSTALL_PREFIX="$ARTIFACT_DIR"

make -j$(nproc)

cp "$BUILD_DIR"/lib/*.so "$ARTIFACT_DIR/" 2>/dev/null || true

echo "ArgoDSM built .so lib files are in $ARTIFACT_DIR."