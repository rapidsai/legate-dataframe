#!/bin/sh
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

# Unfortunately, even though it is split, this steps builds so needs all requirements
echo "installing C++ library and wheel"
cmake --install cpp/build --component testing
pip install python/legate_dataframe*.whl --no-deps
