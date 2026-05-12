"""Full-batch AdamW training loop with log-spaced checkpointing."""
import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from .config import GrokConfig
from .data import cross_entropy_high_precision, model_forward_logits, evaluate


DEFAULT_RUN_ROOT = os.path.join('.', 'grokking_runs')


def log_spaced_epochs(num_epochs: int, num_points: int) -> set:
    """Roughly log-spaced integer epoch indices in [0, num_epochs-1], plus endpoints."""
    if num_epochs <= 1:
        return {0}
    pts = np.unique(np.round(np.geomspace(1, num_epochs, num_points)).astype(int))
    pts = np.clip(pts - 1, 0, num_epochs - 1)
    return set(pts.tolist()) | {0, num_epochs - 1}


def train_loop(model, train_data, train_labels, test_data, test_labels,
               config: GrokConfig, num_epochs: int, run_name: str,
               is_transformer: bool, run_root: str = DEFAULT_RUN_ROOT):
    """Full-batch AdamW training. Records train/test loss+acc every epoch (needed
    for the grokking visualisation) and saves model checkpoints at log-spaced
    epochs to `<run_root>/<run_name>/checkpoints/`. A tqdm progress bar shows ETA
    and current train/test metrics in the postfix.
    """
    optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                            weight_decay=config.weight_decay, betas=config.betas)
    metrics = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    run_dir = os.path.join(run_root, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    save_epochs = log_spaced_epochs(num_epochs, config.num_checkpoints)
    postfix_every = max(1, num_epochs // 500)

    torch.save({'epoch': -1, 'model_state': model.state_dict()},
               os.path.join(ckpt_dir, 'epoch_init.pt'))

    pbar = tqdm(range(num_epochs), desc=run_name, dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        logits = model_forward_logits(model, train_data, is_transformer)
        train_loss = cross_entropy_high_precision(logits, train_labels)
        with torch.no_grad():
            train_acc = (logits.argmax(dim=-1) == train_labels).float().mean().item()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        test_loss, test_acc = evaluate(model, test_data, test_labels, is_transformer)
        train_loss_val = train_loss.item()
        metrics['train_loss'].append(train_loss_val)
        metrics['train_acc'].append(train_acc)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)

        if epoch in save_epochs:
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       os.path.join(ckpt_dir, f'epoch_{epoch:07d}.pt'))

        if epoch % postfix_every == 0 or epoch == num_epochs - 1:
            pbar.set_postfix(
                train_loss=f'{train_loss_val:.2e}',
                test_loss=f'{test_loss:.2e}',
                train_acc=f'{train_acc:.3f}',
                test_acc=f'{test_acc:.3f}',
            )

    pbar.close()

    with open(os.path.join(run_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    with open(os.path.join(run_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    print(f'Saved {len(save_epochs)} checkpoints + metrics to {run_dir}')
    return metrics
