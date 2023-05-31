
rm -rf /mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/results.csv
for i in {0..4}
do
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold$i
GPU=$1
TOPK=1
LR=5e-5
CONS_ALPHA=0.01
SCS_ALPHA=0.01
DROP=0.1
MEMORY=32
OUTMODALITY=SMILES+Graph

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/drug_combo_eval_modalities_comb/binary_class_inf.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/drug_combo_eval_modalities_comb/ \
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
    --num-classes 125 \
    --fp16 \
    --out-modal=$OUTMODALITY \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/drug_name.dict \
    --raw-data-path=data/drug_combo/drug_modalities.pkl \
    --drug-target-path=data/drug_combo/drug_target.csv \
    --no-progress-bar \
    --valid-subset 'test' \
    --skip-update-state-dict \
#############################################################################################################
OUTMODALITY=SMILES+Graph+Drug_target

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD

mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/drug_combo_eval_modalities_comb/binary_class_inf.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/drug_combo_eval_modalities_comb/ \
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
    --num-classes 125 \
    --fp16 \
    --out-modal=$OUTMODALITY \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/drug_name.dict \
    --raw-data-path=data/drug_combo/drug_modalities.pkl \
    --drug-target-path=data/drug_combo/drug_target.csv \
    --no-progress-bar \
    --valid-subset 'test' \
    --skip-update-state-dict \
#############################################################################################################
OUTMODALITY=SMILES+Graph+Drug_target+Text

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD

mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/drug_combo_eval_modalities_comb/binary_class_inf.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/drug_combo_eval_modalities_comb/ \
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
    --num-classes 125 \
    --fp16 \
    --out-modal=$OUTMODALITY \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/drug_name.dict \
    --raw-data-path=data/drug_combo/drug_modalities.pkl \
    --drug-target-path=data/drug_combo/drug_target.csv \
    --no-progress-bar \
    --valid-subset 'test' \
    --skip-update-state-dict \
#############################################################################################################
OUTMODALITY=SMILES+Graph+Drug_target+Text+Drug_Ontology

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD

mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/drug_combo_eval_modalities_comb/binary_class_inf.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/drug_combo_eval_modalities_comb/ \
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
    --num-classes 125 \
    --fp16 \
    --out-modal=$OUTMODALITY \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/drug_name.dict \
    --raw-data-path=data/drug_combo/drug_modalities.pkl \
    --drug-target-path=data/drug_combo/drug_target.csv \
    --no-progress-bar \
    --valid-subset 'test' \
    --skip-update-state-dict \
#############################################################################################################
OUTMODALITY=SMILES+Graph+Drug_target+Text+Drug_Ontology+Side_effect

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD

mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/drug_combo_eval_modalities_comb/binary_class_inf.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/drug_combo_eval_modalities_comb/ \
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
    --num-classes 125 \
    --fp16 \
    --out-modal=$OUTMODALITY \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/drug_name.dict \
    --raw-data-path=data/drug_combo/drug_modalities.pkl \
    --drug-target-path=data/drug_combo/drug_target.csv \
    --no-progress-bar \
    --valid-subset 'test' \
    --skip-update-state-dict \
#############################################################################################################
OUTMODALITY=SMILES+Graph+Drug_target+Text+Drug_Ontology+Side_effect+3D

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD

mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/drug_combo_eval_modalities_comb/binary_class_inf.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/drug_combo_eval_modalities_comb/ \
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
    --num-classes 125 \
    --fp16 \
    --out-modal=$OUTMODALITY \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/drug_name.dict \
    --raw-data-path=data/drug_combo/drug_modalities.pkl \
    --drug-target-path=data/drug_combo/drug_target.csv \
    --no-progress-bar \
    --valid-subset 'test' \
    --skip-update-state-dict \
#############################################################################################################
OUTMODALITY=SMILES+Graph+Drug_target+Text+Drug_Ontology+Side_effect+3D+Drug_Sensitivity_NCI60

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_modalities_comb/$OUTMODALITY/$DATAFOLD

mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/drug_combo_eval_modalities_comb/binary_class_inf.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/drug_combo_eval_modalities_comb/ \
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
    --num-classes 125 \
    --fp16 \
    --out-modal=$OUTMODALITY \
    --drug-dict-path=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/drug_name.dict \
    --raw-data-path=data/drug_combo/drug_modalities.pkl \
    --drug-target-path=data/drug_combo/drug_target.csv \
    --no-progress-bar \
    --valid-subset 'test' \
    --skip-update-state-dict
done