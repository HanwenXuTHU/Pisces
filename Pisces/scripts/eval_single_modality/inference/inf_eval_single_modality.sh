
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold0
GPU=$1
TOPK=1
LR=5e-5
CONS_ALPHA=0.00
SCS_ALPHA=0.00
DROP=0.1
MEMORY=32
OUTMODALITY=SMILES

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/eval_single_modality/get_preds.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/eval_single_modality/ \
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
#####################################################
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold0
GPU=$1
TOPK=1
LR=5e-5
CONS_ALPHA=0.00
SCS_ALPHA=0.00
DROP=0.1
MEMORY=32
OUTMODALITY=Graph

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/eval_single_modality/get_preds.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/eval_single_modality/ \
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
#####################################################
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold0
GPU=$1
TOPK=1
LR=5e-5
CONS_ALPHA=0.00
SCS_ALPHA=0.00
DROP=0.1
MEMORY=32
OUTMODALITY=Text

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/eval_single_modality/get_preds.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/eval_single_modality/ \
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
#####################################################
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold0
GPU=$1
TOPK=1
LR=5e-5
CONS_ALPHA=0.00
SCS_ALPHA=0.00
DROP=0.1
MEMORY=32
OUTMODALITY=Drug_target

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/eval_single_modality/get_preds.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/eval_single_modality/ \
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
#####################################################
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold0
GPU=$1
TOPK=1
LR=5e-5
CONS_ALPHA=0.00
SCS_ALPHA=0.00
DROP=0.1
MEMORY=32
OUTMODALITY=3D

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/eval_single_modality/get_preds.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/eval_single_modality/ \
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
#####################################################
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold0
GPU=$1
TOPK=1
LR=5e-5
CONS_ALPHA=0.00
SCS_ALPHA=0.00
DROP=0.1
MEMORY=32
OUTMODALITY=Drug_Ontology

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/eval_single_modality/get_preds.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/eval_single_modality/ \
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
#####################################################
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold0
GPU=$1
TOPK=1
LR=5e-5
CONS_ALPHA=0.00
SCS_ALPHA=0.00
DROP=0.1
MEMORY=32
OUTMODALITY=Side_effect

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/eval_single_modality/get_preds.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/eval_single_modality/ \
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
#####################################################
TASK=binary_class_task
ARCH=pisces_base
CLSHEAD=heads_classify
CRITERION=multi_modalities_loss
DATAFOLD=fold0
GPU=$1
TOPK=1
LR=5e-5
CONS_ALPHA=0.00
SCS_ALPHA=0.00
DROP=0.1
MEMORY=32
OUTMODALITY=Drug_Sensitivity_NCI60

# DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/transductive/$DATAFOLD/data-bin
LOADDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD
SAVEDIR=/mnt/hanoverdev/scratch/hanwen/exp/Pisces/gdsc_trans_eval_single_modality/$OUTMODALITY/$DATAFOLD

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=$GPU python Pisces/src/eval_single_modality/get_preds.py  $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir Pisces/src/eval_single_modality/ \
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
#####################################################