import os

from .wan_t2v_14B import t2v_14B

os.environ["TOKENIZERS_PARALLELISM"] = "false"

WAN_CONFIGS = {"t2v-14B": t2v_14B}

__all__ = ["WAN_CONFIGS"]
