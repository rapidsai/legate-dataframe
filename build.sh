#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# legate_dataframe build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd "$(dirname "$0")"; pwd)

VALIDARGS="clean liblegate_dataframe legate_dataframe test -v -g -n -s --ptds -h -e"
HELP="$0 [clean] [liblegate_dataframe] [legate_dataframe] [legate] [-v] [-g] [-n] [-s] [--ptds] [--cmake-args=\"<args>\"] [-h]
   clean                       - remove all existing build artifacts and configuration (start over)
   liblegate_dataframe         - build and install the liblegate_dataframe C++ code (tests are build only)
   legate_dataframe            - build and install the legate_dataframe Python package
   test                        - test all of legate-dataframe (requires valgrind and compute-sanitizer)
   -v                          - verbose build mode
   -e                          - build in editable mode (python only)
   -g                          - build for debug
   -n                          - no install step, also applies to python
   --cmake-args=\\\"<args>\\\" - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   -h                          - print this text
   default action (no args) is to build and install 'liblegate_dataframe' and 'legate_dataframe' targets
"
LIBLEGATE_DATAFRAME_BUILD_DIR=${LIBLEGATE_DATAFRAME_BUILD_DIR:=${REPODIR}/cpp/build}
LEGATE_DATAFRAME_BUILD_DIR="${REPODIR}/python/build ${REPODIR}/python/_skbuild"
LEGATE_BUILD_DIR="${REPODIR}/legate/build ${REPODIR}/legate/_skbuild"
BUILD_DIRS="${LIBLEGATE_DATAFRAME_BUILD_DIR} ${LEGATE_DATAFRAME_BUILD_DIR} ${LEGATE_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
EDITABLE_FLAG=""
BUILD_TYPE=Release
INSTALL_TARGET=install
RAN_CMAKE=0

# Set defaults for vars that may not have been defined externally
# If INSTALL_PREFIX is not set, check PREFIX, then check
# CONDA_PREFIX, then fall back to install inside of $LIBLEGATE_DATAFRAME_BUILD_DIR
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX:=$LIBLEGATE_DATAFRAME_BUILD_DIR/install}}}
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo $ARGS | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo $ARGS | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo $ARGS | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo $EXTRA_CMAKE_ARGS | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi
}


if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    # Check for cmake args
    cmakeArgs
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option or formatting, check --help: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG=-v
    set -x
fi
if hasArg -e; then
    EDITABLE_FLAG=--editable
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
        if [ -d "${bd}" ]; then
            find "${bd}" -mindepth 1 -delete
            rmdir "${bd}" || true
        fi
    done

    # Cleaning up python artifacts
    find ${REPODIR}/python/ | grep -E "(__pycache__|\.pyc|\.pyo|\.so|\_skbuild$)"  | xargs rm -rf
fi

# Runs cmake if it has not been run already for build directory
# LIBLEGATE_DATAFRAME_BUILD_DIR
function ensureCMakeRan {
    mkdir -p "${LIBLEGATE_DATAFRAME_BUILD_DIR}"
    cd ${REPODIR}/cpp
    if (( RAN_CMAKE == 0 )); then
        echo "Executing cmake for liblegate_dataframe..."
        cmake -B "${LIBLEGATE_DATAFRAME_BUILD_DIR}" -S . \
              -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
              -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
              ${EXTRA_CMAKE_ARGS}
        RAN_CMAKE=1
    fi
}

################################################################################
# Configure, build, and install liblegate_dataframe
if (( NUMARGS == 0 )) || hasArg liblegate_dataframe; then
    ensureCMakeRan
    echo "building liblegate_dataframe..."
    cmake --build "${LIBLEGATE_DATAFRAME_BUILD_DIR}" -j${PARALLEL_LEVEL} ${VERBOSE_FLAG}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        echo "installing liblegate_dataframe..."
        cmake --build "${LIBLEGATE_DATAFRAME_BUILD_DIR}" -j${PARALLEL_LEVEL} --target install ${VERBOSE_FLAG}
    fi
fi

# Build and install the legate_dataframe Python package
if (( NUMARGS == 0 )) || hasArg legate_dataframe; then
    export INSTALL_PREFIX
    echo "building legate_dataframe..."
    cd ${REPODIR}/python/

    if [[ ${INSTALL_TARGET} != "" ]]; then
        PIP_COMMAND=install
    else
        PIP_COMMAND=wheel
    fi
    SKBUILD_CONFIGURE_OPTIONS="-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} -DCMAKE_LIBRARY_PATH=${LIBLEGATE_DATAFRAME_BUILD_DIR} ${EXTRA_CMAKE_ARGS}" \
        SKBUILD_BUILD_OPTIONS="-j${PARALLEL_LEVEL:-1}" \
        python -m pip ${PIP_COMMAND} --no-build-isolation --no-deps --config-settings rapidsai.disable-cuda=true ${VERBOSE_FLAG} ${EDITABLE_FLAG} .
fi


# Run test
if hasArg test; then
    cd ${REPODIR}

    nvidia-smi

    echo "testing C++..."
    ./ci/run_ctests.sh

    echo "testing Python..."
    ./ci/run_pytests.sh

    echo "build and test downstream third-party project..."
    ./ci/run_thirdparty_example.sh

    # echo "testing C++ with valgrind..."
    # cd ${LIBLEGATE_DATAFRAME_BUILD_DIR}
    # valgrind --suppressions=${REPODIR}/valgrind.suppressions ./gtests/cpp_tests

    # TODO: build legate.core with `./install.py --cuda --debug-sanitizer` and
    #       enable compute-sanitizer.
    #
    # echo "testing C++ compute-sanitizer --tool memcheck..."
    # # TODO: enable "compute-sanitizer --tool memcheck" for all tests by using
    # #       the --suppressions argument from CTK v12.3
    # compute-sanitizer --tool memcheck ./gtests/cpp_tests --gtest_filter=-ParquetTest*
    #
    # # TODO: enable "compute-sanitizer --tool racecheck" for all tests by using
    # #       the --suppressions argument from CTK v12.3
    # echo "testing C++ compute-sanitizer --tool racecheck..."
    # compute-sanitizer --tool racecheck ./gtests/cpp_tests --gtest_filter=-ParquetTest*
fi
