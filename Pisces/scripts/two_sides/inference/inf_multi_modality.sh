TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=multi_heads_topk
CRITERION=multi_modalities_loss
DATAFOLD=fold$1
TOPK=$2
LR=1e-4
CONS_ALPHA=0.01
SCS_ALPHA=0.01
DROP=0.1
MEMORY=32

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/data/xuhw/data/Pisces/twosides/$DATAFOLD/data-bin
LOADDIR=/data/xuhw/exp/Pisces/multi_modal_twosides/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/pisces-topk$TOPK-lr$LR-norm-drop$DROP-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw
SAVEDIR=/data/xuhw/exp/Pisces/multi_modal_twosides/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/pisces-topk$TOPK-lr$LR-norm-drop$DROP-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=1,2,3 python Pisces/src/two_sides/binary_class_inf.py $DATADIR \
    --user-dir Pisces/src/two_sides/ \
    --tensorboard-logdir $SAVEDIR \
    --restore-file $LOADDIR/checkpoint_last.pt \
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
    --top-k $TOPK \
    --drug-dict-path=/data/xuhw/data/Pisces/twosides/$DATAFOLD/drug_name.dict \
    --num-classes 963 \
    --n-memory $MEMORY \
    --consis-alpha $CONS_ALPHA \
    --scores-alpha $SCS_ALPHA \
    --dropout $DROP --attention-dropout $DROP --pooler-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates 5 --total-num-update 10  --max-update 10 \
    --log-format 'simple' --log-interval 100 \
    --fp16 \
    --best-checkpoint-metric bacc --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --valid-subset 'valid' \
    --skip-update-state-dict \