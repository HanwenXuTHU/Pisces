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
from typing import Dict, Optional, Any, List, Tuple, Callable

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
import pickle
import collections

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

def generate_uniform_arrays_between(A, B, num_arrays=50):
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape.")
        
    arrays = []
    for _ in range(num_arrays):
        rand_weights = np.random.rand(*A.shape)
        new_array = A + (B - A) * rand_weights
        arrays.append(new_array)
    arrays = np.array(arrays).squeeze()
    return arrays

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
    emb_list = []

    pair_list, cell_list, time_list = [], [], []
    
    # pdb.set_trace()
    for i, sample in enumerate(tqdm(progress)):
        if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
            break
        with torch.no_grad():
            model.eval()
            criterion.eval()
            sample, _ = trainer._prepare_sample(sample)
            embs, preds, targets, cls_type = task.get_embs(sample, trainer.model, trainer.criterion)
            preds_list.append(preds.squeeze().cpu().numpy())
            targets_list.append(targets)
            emb_list.append(embs)

            pair_list.append(sample['pair'].cpu().numpy())
            cell_list.append(sample['model'].cpu().numpy().reshape(-1))
            time_list.append(sample['time'].cpu().numpy().reshape(-1))

    predic = np.concatenate(preds_list)
    y_test = np.concatenate(targets_list)
    embeddings = np.concatenate(emb_list)

    mean_emb, std_emb = np.mean(embeddings, axis=0), np.std(embeddings, axis=0)
    sim_emb_number = 2000
    cls_head = model.classification_heads['heads_classify'].cuda()
    with torch.no_grad():
        # randomly generate 20000 embeddings using uniform distribution
        sim_emb = np.random.normal(loc=mean_emb, scale=std_emb, size=(sim_emb_number, mean_emb.shape[0]))
        sim_emb = torch.from_numpy(sim_emb).half().cuda()
        sim_emb = sim_emb.unsqueeze(1)
        # repeat 64 times
        sim_emb = sim_emb.repeat(1, 64, 1)
        print(f"sim_emb shape: {sim_emb.shape}")
        preds = cls_head.classifier_2(sim_emb).squeeze()
        print(f"preds shape: {preds.shape}")
        preds_wta = cls_head.wta_layer(preds)
        print(f"preds_wta shape: {preds_wta.shape}")
        sim_emb = sim_emb.cpu().numpy()
        sim_emb = sim_emb[:, 0, :]
        preds_wta = preds_wta.cpu().numpy()
    
    simulate_embs = collections.OrderedDict()
    simulate_embs['emb'] = sim_emb
    simulate_embs['pred'] = preds_wta
    
    emb_path = '/'.join(cfg.task.data.split('/')[0: -2])
    fold_i = cfg.task.data.split('/')[-2]
    emb_path = os.path.join(emb_path, f"simulate_embs_{cfg.model['_name']}.pkl")
    if os.path.exists(emb_path):
        emb_dict = load_obj(emb_path)
    else:
        emb_dict = collections.OrderedDict()
        
    emb_dict[fold_i] = simulate_embs
    save_obj(emb_dict, emb_path)
    print(f"Saving embeddings")


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
