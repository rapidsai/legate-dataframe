#!/bin/sh
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

echo "building legate-dataframe library and wheel"
./build.sh legate_dataframe liblegate_dataframe -n
