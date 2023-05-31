TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
TOPK=8
LR=1e-4
CONS_ALPHA=0.01
SCS_ALPHA=0.00
DROP=0.1
MEMORY=32

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/mnt/hanoverdev/scratch/hanwen/data/Pisces/ddi1000_50_ddi_types/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/ddi_1000/$TASK/$ARCH/$CRITERION/$CLSHEAD/pisces-topk$TOPK-lr$LR-norm-drop$DROP-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/ddi_1000/$TASK/$ARCH/$CRITERION/$CLSHEAD/pisces-topk$TOPK-lr$LR-norm-drop$DROP-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=0,1,2,3 python Pisces/src/ddi_1000/binary_class_inf.py $DATADIR \
    --user-dir Pisces/src/ddi_1000/ \
    --reset-dataloader \
    --restore-file /home/swang/xuhw/research-projects/exp/Pisces/multi_modal_drugbank_trans/binary_class_task/pisces_base/multi_modalities_loss/fold0/heads_classify/pisces-topk8-lr1e-4-norm-drop0.1-memory32-alpha0.01-scsalpha0.00-v2.9-noSMcons-raw/checkpoint_best.pt \
    --ddp-backend=legacy_ddp \
    -s 'a' -t 'b' \
    --datatype 'tg' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 256 \
    --batch-size 256 \
    --optimizer adam \
    --gnn-norm layer \
    --top-k=$TOPK \
    --classification-head-name $CLSHEAD \
    --drug-dict-path=/mnt/hanoverdev/scratch/hanwen/data/Pisces/ddi1000/drug_name.dict \
    --drug-target-path=data/ddi_1000/drug_target.csv \
    --raw-data-path=data/ddi_1000/drug_modalities.pkl \
    --gnn-norm layer \
    --num-classes 86 \
    --fp16 \
    --no-progress-bar \
    --valid-subset 'valid' \