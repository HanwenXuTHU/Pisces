
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold$1
GPU=$2
TOPK=1
LR=5e-5
CONS_ALPHA=0.01
SCS_ALPHA=0.01
DROP=0.1
MEMORY=32
PREOUTMODALITY=SMILES+Graph
OUTMODALITY=SMILES+Graph+Drug_target

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$PREOUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/drug_combo_eval_modalities_comb/train.py $DATADIR \
    --user-dir Pisces/src/drug_combo_eval_modalities_comb/ \
    --tensorboard-logdir $SAVEDIR \
    --reset-optimizer --reset-dataloader --reset-meters \
    --restore-file $LOADDIR/checkpoint_best.pt \
    --ddp-backend=legacy_ddp \
    -s 'a' -t 'b' \
    --datatype 'tg' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 256 \
    --batch-size 128 \
    --update-freq 1 \
    --required-batch-size-multiple 1 \
    --classification-head-name $CLSHEAD \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/drug_name.dict \
    --raw-data-path=data/drug_combo/drug_modalities.pkl \
    --drug-target-path=data/drug_combo/drug_target.csv \
    --out-modal=$OUTMODALITY \
    --num-classes 125 \
    --n-memory $MEMORY \
    --consis-alpha $CONS_ALPHA \
    --scores-alpha $SCS_ALPHA \
    --top-k $TOPK \
    --dropout $DROP --attention-dropout $DROP --pooler-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --fp16 \
    --no-epoch-checkpoints \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates 4000 --total-num-update 100000  --max-update 50000 \
    --log-format 'simple' --log-interval 100 \
    --best-checkpoint-metric bacc --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --save-dir $SAVEDIR | tee -a $SAVEDIR/train.log \
