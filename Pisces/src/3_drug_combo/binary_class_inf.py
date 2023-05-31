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
import pandas as pd
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


def compute_metrics(y_test, pred1, model_1):
    auc_roc1 = sk_metrics.roc_auc_score(y_test, model_1)
    f1_score1 = sk_metrics.f1_score(y_test, pred1)
    p, r, t = sk_metrics.precision_recall_curve(y_test, model_1)
    auc_prc1 = sk_metrics.auc(r, p)
    bacc1 = sk_metrics.balanced_accuracy_score(y_test, pred1)
    kappa1 = sk_metrics.cohen_kappa_score(y_test, pred1)
    return auc_roc1, auc_prc1, f1_score1, bacc1, kappa1


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
    raw_data_dir = 'data/drug_combo'
    drug_smiles_dir = os.path.join(raw_data_dir, 'drug_smiles.csv')
    cell_idxes_dir = os.path.join(raw_data_dir, 'cell_tpm.csv')
    cell_names = pd.read_csv(cell_idxes_dir, index_col=0)['cell_line_names']
    CELL_TO_INDEX_DICT = {cell_names[idx]: idx for idx in range(len(cell_names))}
    all_drug_dict = {}
    df_drug_smiles = pd.read_csv(drug_smiles_dir, index_col=0)
    for idx, smiles in zip(tqdm(df_drug_smiles['drug_names']), df_drug_smiles['smiles']):
        all_drug_dict[idx] = smiles
    drug2id = {}
    id = 0
    for idx, smiles in zip(tqdm(df_drug_smiles['drug_names']), df_drug_smiles['smiles']):
        drug2id[idx] = id
        id += 1
    id2drug = {v: k for k, v in drug2id.items()}
    id2cell = {v: k for k, v in CELL_TO_INDEX_DICT.items()}
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
    preds_ab_list, preds_ac_list, preds_bc_list = [], [], []
    targets_list = []

    cls_list = []
    drug_a_list = []
    drug_b_list = []
    drug_c_list = []
    
    # pdb.set_trace()
    for i, sample in enumerate(tqdm(progress)):
        if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
            break
        with torch.no_grad():
            model.eval()
            criterion.eval()
            sample, _ = trainer._prepare_sample(sample)
            preds, targets, cls_type = task.ddi_inference_step(sample, trainer.model, trainer.criterion)
            preds_ab_list.append(preds[0])
            preds_ac_list.append(preds[1])
            preds_bc_list.append(preds[2])
            cls_list.append(cls_type)
            targets_list.append(targets)
            drug_a_list.append(sample['pairab'].cpu().numpy()[:, 0])
            drug_b_list.append(sample['pairab'].cpu().numpy()[:, 1])
            drug_c_list.append(sample['pairac'].cpu().numpy()[:, 1])
    cls_list = np.concatenate(cls_list, axis=0).reshape(-1)
    drug_a_list = np.concatenate(drug_a_list, axis=0)
    drug_b_list = np.concatenate(drug_b_list, axis=0)
    drug_c_list = np.concatenate(drug_c_list, axis=0)
    predic_ab = np.concatenate(preds_ab_list)
    predic_ac = np.concatenate(preds_ac_list)
    predic_bc = np.concatenate(preds_bc_list)

    model_1 = (predic_ab + predic_ac + predic_bc) / 3
    model_2 = (predic_ab + predic_ac + predic_bc + \
               np.multiply(predic_ab, predic_ac)**0.5 + \
               np.multiply(predic_ab, predic_bc)**0.5 + \
               np.multiply(predic_ac, predic_bc)**0.5) / 6
    model_3 = (predic_ab + predic_ac + predic_bc + \
               np.multiply(predic_ab, predic_ac)**0.5 + \
               np.multiply(predic_ab, predic_bc)**0.5 + \
               np.multiply(predic_ac, predic_bc)**0.5 + 
               np.multiply(np.multiply(predic_ab, predic_ac), predic_bc)**(1/3)) / 7
    y_test = np.concatenate(targets_list)
    pred1 = (model_1 >= 0.5).astype(np.int64)
    pred2 = (model_2 >= 0.5).astype(np.int64)
    pred3 = (model_3 >= 0.5).astype(np.int64)

    auc_roc1, auc_prc1, f1_score1, bacc1, kappa1 = compute_metrics(y_test, pred1, model_1)
    auc_roc2, auc_prc2, f1_score2, bacc2, kappa2 = compute_metrics(y_test, pred2, model_2)
    auc_roc3, auc_prc3, f1_score3, bacc3, kappa3 = compute_metrics(y_test, pred3, model_3)

    print(f"bacc: {bacc1:.4f}, auc_roc: {auc_roc1:.4f}, auc_prc: {auc_prc1:.4f}, f1_score: {f1_score1:.4f}, kappa: {kappa1:.4f}")
    print(f"bacc: {bacc2:.4f}, auc_roc: {auc_roc2:.4f}, auc_prc: {auc_prc2:.4f}, f1_score: {f1_score2:.4f}, kappa: {kappa2:.4f}")
    print(f"bacc: {bacc3:.4f}, auc_roc: {auc_roc3:.4f}, auc_prc: {auc_prc3:.4f}, f1_score: {f1_score3:.4f}, kappa: {kappa3:.4f}")
    
    save_dir = os.path.dirname(cfg.task.data)
    file_name = os.path.join(save_dir, 'results.txt')
    with open(file_name, 'w') as f:
        #f.write(f"bacc: {bacc}, auc_prc: {auc_prc}, f1_score: {f1_score}, kappa: {kappa}")
        # save in 4 decimal places
        f.write(f"bacc1: {bacc1:.4f}, auc_roc1: {auc_roc1:.4f}, auc_prc1: {auc_prc1:.4f}, f1_score1: {f1_score1:.4f}, kappa1: {kappa1:.4f}\n")
        f.write(f"bacc2: {bacc2:.4f}, auc_roc2: {auc_roc2:.4f}, auc_prc2: {auc_prc2:.4f}, f1_score2: {f1_score2:.4f}, kappa2: {kappa2:.4f}\n")
        f.write(f"bacc3: {bacc3:.4f}, auc_roc3: {auc_roc3:.4f}, auc_prc3: {auc_prc3:.4f}, f1_score3: {f1_score3:.4f}, kappa3: {kappa3:.4f}\n")

    model1_pred_dir = os.path.join(save_dir, 'model1_pred.csv')
    with open(model1_pred_dir, 'w') as f:
        f.write('drug_a,drug_b,drug_c,cell_line,pred,target\n')
        for i in range(len(model_1)):
            if drug_c_list[i] == drug_b_list[i] or drug_c_list[i] == drug_a_list[i]:
                continue
            f.write(f'{id2drug[drug_a_list[i]]},{id2drug[drug_b_list[i]]},{id2drug[drug_c_list[i]]},{id2cell[cls_list[i]]},{model_1[i]},{y_test[i]}\n')
    
    model1_cell_dir = os.path.join(save_dir, 'model1_cell.csv')
    model2_cell_dir = os.path.join(save_dir, 'model2_cell.csv')
    model3_cell_dir = os.path.join(save_dir, 'model3_cell.csv')
    m1_c = open(model1_cell_dir, 'w')
    m2_c = open(model2_cell_dir, 'w')
    m3_c = open(model3_cell_dir, 'w')
    m1_c.write('cell_line, auc_roc, auc_prc, f1_score, bacc, kappa\n')
    m2_c.write('cell_line, auc_roc, auc_prc, f1_score, bacc, kappa\n')
    m3_c.write('cell_line, auc_roc, auc_prc, f1_score, bacc, kappa\n')

    for i in range(125):
        if len(pred1[cls_list == i]) < 1 or np.max(y_test[cls_list == i]) < 1:
            continue
        y_test_curr = y_test[cls_list == i]
        pred1_curr = pred1[cls_list == i]
        pred2_curr = pred2[cls_list == i]
        pred3_curr = pred3[cls_list == i]
        model_1_curr = model_1[cls_list == i]
        model_2_curr = model_2[cls_list == i]
        model_3_curr = model_3[cls_list == i]

        auc_roc1, auc_prc1, f1_score1, bacc1, kappa1 = compute_metrics(y_test_curr, pred1_curr, model_1_curr)
        auc_roc2, auc_prc2, f1_score2, bacc2, kappa2 = compute_metrics(y_test_curr, pred2_curr, model_2_curr)
        auc_roc3, auc_prc3, f1_score3, bacc3, kappa3 = compute_metrics(y_test_curr, pred3_curr, model_3_curr)
        
        m1_c.write(f"{id2cell[i]}, {auc_roc1:.4f}, {auc_prc1:.4f}, {f1_score1:.4f}, {bacc1:.4f}, {kappa1:.4f}\n")
        m2_c.write(f"{id2cell[i]}, {auc_roc2:.4f}, {auc_prc2:.4f}, {f1_score2:.4f}, {bacc2:.4f}, {kappa2:.4f}\n")
        m3_c.write(f"{id2cell[i]}, {auc_roc3:.4f}, {auc_prc3:.4f}, {f1_score3:.4f}, {bacc3:.4f}, {kappa3:.4f}\n")
    
    model1_drug_dir = os.path.join(save_dir, 'model1_drug.csv')
    model2_drug_dir = os.path.join(save_dir, 'model2_drug.csv')
    model3_drug_dir = os.path.join(save_dir, 'model3_drug.csv')
    m1_d = open(model1_drug_dir, 'w')
    m2_d = open(model2_drug_dir, 'w')
    m3_d = open(model3_drug_dir, 'w')
    m1_d.write('drug,auc_roc,auc_prc,f1_score,bacc,kappa\n')
    m2_d.write('drug,auc_roc,auc_prc,f1_score,bacc,kappa\n')
    m3_d.write('drug,auc_roc,auc_prc,f1_score,bacc,kappa\n')

    for i in id2drug.keys():
        sel_area = np.logical_or(drug_a_list == i, drug_b_list == i)
        sel_area = np.logical_or(sel_area, drug_c_list == i)
        if len(pred1[sel_area]) == 0 or np.max(y_test[sel_area]) < 1:
            continue
        y_test_curr = y_test[sel_area]
        pred1_curr = pred1[sel_area]
        pred2_curr = pred2[sel_area]
        pred3_curr = pred3[sel_area]
        model_1_curr = model_1[sel_area]
        model_2_curr = model_2[sel_area]
        model_3_curr = model_3[sel_area]

        auc_roc1, auc_prc1, f1_score1, bacc1, kappa1 = compute_metrics(y_test_curr, pred1_curr, model_1_curr)
        auc_roc2, auc_prc2, f1_score2, bacc2, kappa2 = compute_metrics(y_test_curr, pred2_curr, model_2_curr)
        auc_roc3, auc_prc3, f1_score3, bacc3, kappa3 = compute_metrics(y_test_curr, pred3_curr, model_3_curr)

        m1_d.write(f"{id2drug[i]}, {auc_roc1:.4f}, {auc_prc1:.4f}, {f1_score1:.4f}, {bacc1:.4f}, {kappa1:.4f}\n")
        m2_d.write(f"{id2drug[i]}, {auc_roc2:.4f}, {auc_prc2:.4f}, {f1_score2:.4f}, {bacc2:.4f}, {kappa2:.4f}\n")
        m3_d.write(f"{id2drug[i]}, {auc_roc3:.4f}, {auc_prc3:.4f}, {f1_score3:.4f}, {bacc3:.4f}, {kappa3:.4f}\n")

    m1_c.close()
    m2_c.close()
    m3_c.close()
    m1_d.close()
    m2_d.close()
    m3_d.close()

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
