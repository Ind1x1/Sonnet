
# sonnet optimizer

import torch
import os
from deepspeed import comm as dist
from packaging import version as pkg_version
from collections import OrderedDict
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from deepspeed.runtime import ZeROOptimizer,SonnetOptimizer
from deepspeed.runtime.fp16.loss_scaler import CreateLossScaler
from deepspeed.runtime.utils import (bwc_tensor_model_parallel_rank, get_global_norm, empty_cache, see_memory_usage,
                                     inf, is_model_parallel_parameter, align_dense_tensors, all_gather_dp_groups)

from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.ops.adam import DeepSpeedCPUAdam,SonnetCPUAdam
from deepspeed.utils import logger
from deepspeed.moe.utils import is_moe_param
from deepspeed.git_version_info import version

from deepspeed.runtime.constants import PIPE_REPLICATED
from deepspeed.accelerator import get_accelerator

from deepspeed.checkpoint.constants import (DS_VERSION, GROUP_PADDINGS, PARTITION_COUNT, LOSS_SCALER,
                                            SINGLE_PARTITION_OF_FP32_GROUPS, BASE_OPTIMIZER_STATE,
                                            BASE_OPTIMIZER_STATE_STEP, CLIP_GRAD, ZERO_STAGE, PARAM_SLICE_MAPPINGS)
from deepspeed.utils import link_hp_params
from deepspeed.checkpoint import enable_universal_checkpoint

from deepspeed.utils import groups
# Toggle this to true to enable correctness test
# with gradient partitioning and without
pg_correctness_test = False

OPTIMIZER_ALLGATHER_TIMER = 'optimizer_allgather'
OPTIMIZER_GRADIENTS_TIMER = 'optimizer_gradients'
OPTIMIZER_STEP_TIMER = 'optimizer_step'
OPTIMIZER_TIMERS = [OPTIMIZER_ALLGATHER_TIMER, OPTIMIZER_GRADIENTS_TIMER, OPTIMIZER_STEP_TIMER]


class SonnetCPUoptimizer(SonnetOptimizer):
    """
    
    """
    def __init__(self,
                 ):
        
        
        