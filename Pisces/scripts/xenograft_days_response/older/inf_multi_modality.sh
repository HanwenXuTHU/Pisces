
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold$1
GPU=$2
TOPK=$3
LR=5e-4
CONS_ALPHA=0.01
SCS_ALPHA=0.01
DROP=0.1
POOLER=0.1
MEMORY=32

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/xenograft_data/response_days_v1/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/xenograft_response_days_v1/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/pisces-fixed-topk$TOPK-lr$LR-norm-drop$DROP-pooler$POOLER-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/xenograft_response_days_v1/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/pisces-fixed-topk$TOPK-lr$LR-norm-drop$DROP-pooler$POOLER-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/xenograft_extrapolation/binary_class_inf.py $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/xenograft_extrapolation/ \
    --ddp-backend=legacy_ddp \
    --reset-dataloader \
    -s 'a' -t 'b' \
    --datatype 'tg' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 256 \
    --batch-size 16 \
    --n-memory $MEMORY \
    --optimizer adam \
    --classification-head-name $CLSHEAD \
    --top-k $TOPK \
    --fp16 \
    --gnn-norm layer \
    --num-classes 399 \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/xenograft_data/best_response/$DATAFOLD/drug_name.dict \
    --raw-data-path=data/xenograft/drug_modalities.pkl \
    --drug-target-path=data/xenograft/drug_target.csv \
    --no-progress-bar \
    --valid-subset 'test' \
    --skip-update-state-dict \