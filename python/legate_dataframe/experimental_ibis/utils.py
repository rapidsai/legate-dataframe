# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def _gen_name(namespace, identifier):
    """Generate a name from namespace and identifier.  Similar to ibis'
    helper, but ensure identical names on all nodes.

    TODO: For no just uses the identifier and namespace

    TODO: In the future, allow identifier to be None (or never use it).
          For that we need to ensure identical randomness on workers.
    """
    return f"ibis_{namespace}_{identifier}"
