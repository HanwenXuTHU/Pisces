TASK=binary_class_ind
ARCH=pisces_small_ind
CLSHEAD=heads_ablation_ind
CRITERION=multi_modalities_loss
DATAFOLD=fold$1
TOPK=$2
LR=1e-4
CONS_ALPHA=0.01
SCS_ALPHA=0.00
DROP=0.3
MEMORY=32
CSTR=0.0

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/drugbank_ind/$DATAFOLD/data-bin
LOADDIR=/home/swang/xuhw/research-projects/exp/Pisces/multi_modal_drugbank_ind/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/pisces-topk$TOPK-cst$CSTR-lr$LR-norm-drop$DROP-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw
SAVEDIR=/home/swang/xuhw/research-projects/exp/Pisces/multi_modal_drugbank_ind/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/pisces-topk$TOPK-cst$CSTR-lr$LR-norm-drop$DROP-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=2,3 python Pisces/src/ddi/train.py $DATADIR \
    --user-dir Pisces/src/ddi/ \
    --tensorboard-logdir $SAVEDIR \
    --restore-file $LOADDIR/checkpoint_last.pt \
    --ddp-backend=legacy_ddp \
    -s 'a' -t 'b' \
    --datatype 'tg' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 200 \
    --batch-size 128 \
    --update-freq 1 \
    --required-batch-size-multiple 1 \
    --classification-head-name $CLSHEAD \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/drugbank_ind/$DATAFOLD/drug_name.dict \
    --drug-target-path=data/drugbank_ddi/drug_target.csv \
    --raw-data-path=data/drugbank_ddi/drug_modalities.pkl \
    --num-classes 86 \
    --n-memory $MEMORY \
    --consis-alpha $CONS_ALPHA \
    --scores-alpha $SCS_ALPHA \
    --top-k=$TOPK \
    --fp16 \
    --gnn-norm layer \
    --dropout $DROP --attention-dropout $DROP --pooler-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates 4000 --total-num-update 50000  --max-update 50000 \
    --log-format 'simple' --log-interval 100 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --save-dir $SAVEDIR | tee -a $SAVEDIR/train.log \
