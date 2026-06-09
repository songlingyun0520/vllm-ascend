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

"""Patch EngineCoreProc to run FractionCalculation before each decode step.

Activated when VLLM_ASCEND_DSA_MTP_ENABLE=1.
Requires the ``patches/`` package to be on PYTHONPATH, e.g.:
    export PYTHONPATH=/vllm-workspace:$PYTHONPATH
"""

import logging

from vllm.v1.engine.core import EngineCoreProc

from patches.dsa_mtp_planner import FractionCalculation

logger = logging.getLogger(__name__)

# CPU pinned-memory float tensor written by FractionCalculation each decode step.
_FRACTION_FLOAT = None

_original_process_engine_step = EngineCoreProc._process_engine_step


def _patched_process_engine_step(self):
    original_schedule = self.scheduler.schedule

    def _wrapped_schedule():
        global _FRACTION_FLOAT
        scheduler_output = original_schedule()

        _FRACTION_FLOAT = FractionCalculation(scheduler_output)

        num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
        batch_size = len(num_scheduled_tokens) if num_scheduled_tokens is not None else 0
        split_ratio = float(_FRACTION_FLOAT.item()) if _FRACTION_FLOAT is not None else 0.0
        logger.info(
            "[DSA-MTP] scheduler_output.num_scheduled_tokens keys=%d  "
            "=> FractionCalculation result: _FRACTION_FLOAT=%.6f",
            batch_size,
            split_ratio,
        )

        return scheduler_output

    self.scheduler.schedule = _wrapped_schedule
    try:
        return _original_process_engine_step(self)
    finally:
        self.scheduler.schedule = original_schedule


EngineCoreProc._process_engine_step = _patched_process_engine_step
logger.info("[DSA-MTP] patch_dsa_mtp_fraction applied: EngineCoreProc._process_engine_step patched.")
