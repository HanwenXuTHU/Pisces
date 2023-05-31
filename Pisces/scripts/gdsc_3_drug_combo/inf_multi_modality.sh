
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
GPU=$1
TOPK=$2
LR=5e-4
CONS_ALPHA=0.01
SCS_ALPHA=0.01
DROP=0.1
POOLER=0.1
MEMORY=32

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/gdsc_3_drug_combo/data-bin

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/3_drug_combo/binary_class_inf.py $DATADIR \
    --restore-file /mnt/hanoverdev/scratch/hanwen/exp/Pisces/xenograft_gdsc_transfer/binary_class_task/pisces_base/multi_modalities_loss/fold0/heads_classify/pisces-topk8-lr5e-5-norm-drop0.1-memory32-alpha0.01-scsalpha0.01-v2.9-noSMcons-raw/checkpoint_best.pt \
    --user-dir Pisces/src/3_drug_combo/ \
    --ddp-backend=legacy_ddp \
    --reset-optimizer --reset-dataloader --reset-meters \
    -s 'a' -t 'b' \
    --datatype 'tg' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 256 \
    --batch-size 128 \
    --n-memory $MEMORY \
    --optimizer adam \
    --classification-head-name $CLSHEAD \
    --top-k $TOPK \
    --fp16 \
    --num-classes 1 \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/gdsc_3_drug_combo/drug_name.dict \
    --raw-data-path=data/drug_combo/drug_modalities.pkl \
    --drug-target-path=data/drug_combo/drug_target.csv \
    --no-progress-bar \
    --valid-subset 'train' \
    --skip-update-state-dict