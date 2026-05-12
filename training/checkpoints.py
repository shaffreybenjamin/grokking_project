"""Loading utilities for trained-model checkpoints."""
import glob
import os
import pickle

import torch

from .config import GrokConfig
from .loop import DEFAULT_RUN_ROOT


class _CompatibleUnpickler(pickle.Unpickler):
    """Re-route `__main__.GrokConfig` (the pickle path used when GrokConfig was
    defined inline in the notebook) to `training.config.GrokConfig`. Lets us load
    config.pkl files saved before the package extraction without re-saving them."""
    def find_class(self, module, name):
        if name == 'GrokConfig' and module in ('__main__', '__mp_main__'):
            return GrokConfig
        return super().find_class(module, name)


def _load_pickle(path: str):
    with open(path, 'rb') as f:
        return _CompatibleUnpickler(f).load()


def load_final_checkpoint(run_name: str, run_root: str = DEFAULT_RUN_ROOT,
                          device=None):
    """Load the highest-numbered epoch_NNNNNNN.pt under `<run_root>/<run_name>/`.

    Returns: (config, checkpoint_dict, metrics) — same shape as the notebook's
    original helper. Caller is responsible for instantiating the model class
    (GrokTransformer / ChughtaiMLP) and applying `load_state_dict(checkpoint['model_state'])`.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = os.path.join(run_root, run_name)

    config = _load_pickle(os.path.join(run_dir, 'config.pkl'))
    metrics = _load_pickle(os.path.join(run_dir, 'metrics.pkl'))

    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_[0-9]*.pt')))
    if not checkpoints:
        raise FileNotFoundError(f'No epoch checkpoints found in {ckpt_dir}')
    final_ckpt = checkpoints[-1]
    print(f'Loading: {final_ckpt}')

    checkpoint = torch.load(final_ckpt, map_location=device)
    print(f'Epoch: {checkpoint["epoch"]}')
    return config, checkpoint, metrics
