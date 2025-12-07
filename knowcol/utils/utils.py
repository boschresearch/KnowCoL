import os
from pathlib import Path
import shutil
import time
from typing import Dict, List, Union
import numpy as np
import torch
import pytorch_lightning
import tqdm


road_sign_index = range(0,43)
color_index = range(43,50)
shape_index = range(50,54)

def get_checkpoints_for_epochs(experiment_folder: Path, epochs: Union[List, str]) -> List:
    if isinstance(epochs, str):
        epochs = epochs.split(",")
        epochs = list(map(int, epochs))
    ep = lambda s: int(s.stem.split("=")[1])
    return [chk for chk in get_all_checkpoints(experiment_folder) if ep(chk) in epochs]


def get_all_checkpoints(experiment_folder: Path) -> List:
    if experiment_folder.is_dir():
        checkpoint_folder = experiment_folder / "saved_models"
        if checkpoint_folder.is_dir():
            checkpoints = sorted(Path(checkpoint_folder).iterdir(), key=lambda chk: chk.stat().st_mtime)
            if len(checkpoints):
                return [chk for chk in checkpoints if chk.suffix == ".ckpt"]
    return []


def get_last_checkpoint(experiment_folder: Path) -> Union[Path, None]:
    # return newest checkpoint according to creation time
    checkpoints = get_all_checkpoints(experiment_folder)
    if len(checkpoints):
        return checkpoints[-1]
    return None

def info_packages() -> Dict[str, str]:
    return {
        "numpy": np.__version__,
        "pyTorch_version": torch.__version__,
        "pyTorch_debug": str(torch.version.debug),
        "pytorch-lightning": pytorch_lightning.__version__,
        "tqdm": tqdm.__version__,
    }

def info_cuda() -> Dict[str, Union[str, List[str]]]:
    return {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        # 'nvidia_driver': get_nvidia_driver_version(run_lambda),
        "available": str(torch.cuda.is_available()),
        "version": torch.version.cuda,
    }
    
def nice_print(details: Dict, level: int = 0) -> List:
    lines = []
    LEVEL_OFFSET = "\t"
    KEY_PADDING = 20
    for k in sorted(details):
        key = f"* {k}:" if level == 0 else f"- {k}:"
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level + 1) * LEVEL_OFFSET + "- " + v for v in details[k]]
        else:
            template = "{:%is} {}" % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [(level * LEVEL_OFFSET) + key_val]
    return lines

def print_system_env_info():
    details = {
        "Packages": info_packages(),
        "CUDA": info_cuda(),
    }
    lines = nice_print(details)
    text = os.linesep.join(lines)
    return text
