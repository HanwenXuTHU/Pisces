
for i in {0..9}
do
TASK=binary_class_task
ARCH=transfer_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold$i
GPU=$1
TOPK=$2
LR=5e-4
CONS_ALPHA=0.01
SCS_ALPHA=0.01
DROP=0.1
POOLER=0.1
MEMORY=32

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/xenograft_data_10fold/response_days_extrapolation/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/xenograft_extrapolation/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/pisces-fixed-topk$TOPK-lr$LR-norm-drop$DROP-pooler$POOLER-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/xenograft_extrapolation/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/pisces-fixed-topk$TOPK-lr$LR-norm-drop$DROP-pooler$POOLER-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/xenograft_extrapolation/train.py $DATADIR \
    --user-dir Pisces/src/xenograft_extrapolation \
    --tensorboard-logdir $SAVEDIR \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file /mnt/hanoverdev/scratch/hanwen/exp/Pisces/xenograft_gdsc_transfer/binary_class_task/pisces_base/multi_modalities_loss/fold0/heads_classify/pisces-topk8-lr5e-5-norm-drop0.1-memory32-alpha0.01-scsalpha0.01-v2.9-noSMcons-raw/checkpoint_best.pt \
    --ddp-backend=legacy_ddp \
    -s 'a' -t 'b' \
    --datatype 'tg' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 256 \
    --batch-size 64 \
    --update-freq 1 \
    --required-batch-size-multiple 1 \
    --classification-head-name $CLSHEAD \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/xenograft_data_10fold/response_days_v2/$DATAFOLD/drug_name.dict \
    --raw-data-path=data/xenograft/drug_modalities.pkl \
    --drug-target-path=data/xenograft/drug_target.csv \
    --num-classes 399 \
    --n-memory $MEMORY \
    --consis-alpha $CONS_ALPHA \
    --scores-alpha $SCS_ALPHA \
    --top-k $TOPK \
    --gnn-norm layer \
    --dropout $DROP --attention-dropout $DROP --pooler-dropout $POOLER \
    --weight-decay 0.01 --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-06 \
    --fp16 \
    --clip-norm 0.0 \
    --no-epoch-checkpoints \
    --lr $LR \
    --max-update 10000 \
    --log-format 'simple' --log-interval 100 \
    --best-checkpoint-metric pearsonr --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --save-dir $SAVEDIR | tee -a $SAVEDIR/train.log
done
