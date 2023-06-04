
rm -rf /home/swang/xuhw/research-projects/data/Pisces/transductive/results_pisces_base.csv
for i in {0..4}
do
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold$i
GPU=$1
TOPK=8
LR=5e-5
CONS_ALPHA=0.01
SCS_ALPHA=0.01
DROP=0.1
MEMORY=32

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/pisces-topk$TOPK-lr$LR-norm-drop$DROP-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/pisces-topk$TOPK-lr$LR-norm-drop$DROP-memory$MEMORY-alpha$CONS_ALPHA-scsalpha$SCS_ALPHA-v2.9-noSMcons-raw

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/drug_combo/binary_class_inf.py $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/drug_combo/ \
    --ddp-backend=legacy_ddp \
    --reset-dataloader \
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
    --wta-linear \
    --num-classes 125 \
    --fp16 \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/drug_name.dict \
    --raw-data-path=data/drug_combo/drug_modalities.pkl \
    --drug-target-path=data/drug_combo/drug_target.csv \
    --no-progress-bar \
    --valid-subset 'test' \
    --skip-update-state-dict
done
