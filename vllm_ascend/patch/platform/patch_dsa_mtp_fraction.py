#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load the DSA-MTP fraction patch from the project patches/ directory.

Activated when VLLM_ASCEND_DSA_MTP_ENABLE=1.
Requires the project root (containing the ``patches/`` package) to be on
PYTHONPATH, e.g.:
    export PYTHONPATH=/path/to/dsa_batch_with_vllm:$PYTHONPATH
"""

import logging

logger = logging.getLogger(__name__)

try:
    import patches.patch_pre_decode_compute  # noqa: F401
except ImportError as exc:
    logger.warning(
        "[DSA-MTP] Could not import patches.patch_pre_decode_compute: %s. "
        "Make sure the project root is on PYTHONPATH "
        "(export PYTHONPATH=/path/to/dsa_batch_with_vllm:$PYTHONPATH).",
        exc,
    )
