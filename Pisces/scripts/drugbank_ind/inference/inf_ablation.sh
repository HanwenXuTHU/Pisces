TASK=binary_class_ind
ARCH=pisces_small_ind
CLSHEAD=heads_ablation_ind
CRITERION=multi_modalities_loss
DATAFOLD=fold$1
GPU=$2
TOPK=$3
LR=1e-4
CONS_ALPHA=0.01
SCS_ALPHA=0.00
DROP=0.3
MEMORY=32
CSTR=0.5

DATADIR=/home/swang/xuhw/research-projects/data/Pisces/drugbank_ind/$DATAFOLD/data-bin
LOADDIR=/home/swang/xuhw/research-projects/exp/Pisces/multi_modal_drugbank_ind/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/bitop-pisces-topk$TOPK-cst$CSTR-lr$LR-norm-drop$DROP-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw
SAVEDIR=/home/swang/xuhw/research-projects/exp/Pisces/multi_modal_drugbank_ind/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/bitop-pisces-topk$TOPK-cst$CSTR-lr$LR-norm-drop$DROP-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/ddi/binary_class_inf.py $DATADIR \
    --user-dir Pisces/src/ddi/ \
    --reset-dataloader \
    --restore-file $LOADDIR/checkpoint_best.pt \
    --ddp-backend=legacy_ddp \
    -s 'a' -t 'b' \
    --datatype 'tg' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 200 \
    --batch-size 128 \
    --optimizer adam \
    --gnn-norm layer \
    --top-k=$TOPK \
    --is-bitop \
    --classification-head-name $CLSHEAD \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/drugbank_trans/$DATAFOLD/drug_name.dict \
    --drug-target-path=data/drugbank_ddi/drug_target.csv \
    --raw-data-path=data/drugbank_ddi/drug_modalities.pkl \
    --gnn-norm layer \
    --num-classes 86 \
    --fp16 \
    --no-progress-bar \
    --valid-subset 'valid' \
    --skip-update-state-dict \