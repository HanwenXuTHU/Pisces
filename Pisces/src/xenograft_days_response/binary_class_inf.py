#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
import pdb
from typing import Dict, Optional, Any, List, Tuple, Callable

from scipy import stats
import numpy as np
import torch

sys.path.append(os.getcwd())

from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics as sk_metrics
from tqdm import tqdm
import pdb

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def spearmanr(y_true, y_pred):
    return stats.spearmanr(y_true, y_pred)[0]

def scatter_plot(x, y, save_name):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, linregress

    # Set Seaborn theme for beautiful plots
    sns.set_theme(style='whitegrid', context='notebook')

    pearson_r, _ = pearsonr(x, y)

    # Calculate the linear regression line
    slope, intercept, _, _, _ = linregress(x, y)
    line = slope * x + intercept

    # Create a scatter plot using the 'viridis' color map
    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, c=y, cmap='viridis', edgecolors='k', s=80, alpha=0.8)

    # Add the linear regression line
    plt.plot(x, line, 'r')
    plt.text(0.05, 0.95, f'Pearson r = {pearson_r:.2f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top')

    # Add labels and title
    plt.xlabel('Predicted response', fontsize=14)
    plt.ylabel('True Response', fontsize=14)
    #plt.title('Beautiful Scatter Plot', fontsize=18)

    # Remove top and right spines for a cleaner look
    sns.despine()

    # Save the figure as a PNG file
    plt.savefig(save_name, dpi=300)

    # Display the plot
    plt.show()

def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    criterion.set_label_id(task.label_reverse_id)

    # logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)
        
    trainer = Trainer(cfg, task, model, criterion)
    checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    
    validate(cfg, trainer, task)


# TODO
def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(0)
    subset = cfg.dataset.valid_subset
    logger.info('begin validation on "{}" subset'.format(subset))

    # Initialize data iterator
    itr = trainer.get_valid_iterator(subset).next_epoch_itr(
        shuffle=False, set_dataset_epoch=False  # use a fixed valid set
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=0,
        prefix=f"valid on '{subset}' subset",
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
    )

    model = trainer.model
    criterion = trainer.criterion
    preds_list = []
    targets_list = []

    cls_list = []
    
    # pdb.set_trace()
    for i, sample in enumerate(tqdm(progress)):
        if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
            break
        with torch.no_grad():
            model.eval()
            criterion.eval()
            sample, _ = trainer._prepare_sample(sample)
            preds, targets, cls_type = task.ddi_inference_step(sample, trainer.model, trainer.criterion)
            preds_list.append(preds.squeeze().cpu().numpy())
            cls_list.append(cls_type)
            targets_list.append(targets)


    predic = np.concatenate(preds_list)
    y_test = np.concatenate(targets_list)
    # pdb.set_trace()
    # from collections import Counter
    # cell_counter = Counter(cls_types.tolist())
    
    # np.save('predic_leave_combs.npy', predic)
    # np.save('y_test_leave_combs.npy', y_test)
    # np.save('cell_types_leave_combs.npy', cls_types)
    
    # np.save('predic_leave_combs_extra.npy', predic)
    # np.save('y_test_leave_combs_extra.npy', y_test)
    # np.save('cell_types_leave_combs_extra.npy', cls_types)

    # np.save('predic_trans_graph.npy', predic)
    # np.save('y_test_trans_graph.npy', y_test)
    # np.save('cell_types_trans_graph.npy', cls_types)

    # np.save('predic_trans_smiles.npy', predic)
    # np.save('y_test_trans_smiles.npy', y_test)
    # np.save('cell_types_trans_smiles.npy', cls_types)
    pearsonr = np.corrcoef(predic, y_test)[0, 1]
    spr = spearmanr(predic, y_test)
    rmse = np.sqrt(mean_squared_error(predic, y_test))

    save_dir = os.path.dirname(cfg.checkpoint.restore_file)
    save_path = f'{save_dir}/scatter_plot.png'
    scatter_plot(predic, y_test, save_path)
    # save images to path
    print('saved scatter plot to: ', save_path)

    print(f"pearsonr: {pearsonr}, spearmanr: {spr}, rmse: {rmse}")
    res_path = '/'.join(cfg.task.data.split('/')[0: -2])
    f = open(os.path.join(res_path, f"res_{cfg.model['_name']}.csv"), 'a')
    f.write(f'{pearsonr},{spr},{rmse}\n')
    f.close()
    
def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}")

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
