
# binarize dataset
# DATADIR=/data/linjc/dds/data/transductive_3fold/$DATAFOLD/
W=32

DATADIR='/home/swang/xuhw/research-projects/data/Pisces/gdsc_3_drug_combo'

python fairseq_cli/preprocess.py \
    -s 'a' -t 'b' \
    --trainpref $DATADIR/train \
    --validpref $DATADIR/valid \
    --destdir  $DATADIR/data-bin/ \
    --srcdict molecule/dict.txt \
    --joined-dictionary \
    --workers $W \
    --molecule \

python fairseq_cli/preprocess.py \
    -s 'a' -t 'c' \
    --trainpref $DATADIR/train \
    --validpref $DATADIR/valid \
    --destdir  $DATADIR/data-bin/ \
    --srcdict molecule/dict.txt \
    --joined-dictionary \
    --workers $W \
    --molecule \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.cell \
    --validpref $DATADIR/valid.cell \
    --destdir $DATADIR/data-bin/cell/ \
    --workers $W --srcdict $DATADIR/cell.dict \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.pairab \
    --validpref $DATADIR/valid.pairab \
    --destdir $DATADIR/data-bin/pairab/ \
    --workers $W --srcdict $DATADIR/drug_id.dict \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.pairac \
    --validpref $DATADIR/valid.pairac \
    --destdir $DATADIR/data-bin/pairac/ \
    --workers $W --srcdict $DATADIR/drug_id.dict \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.pairbc \
    --validpref $DATADIR/valid.pairbc \
    --destdir $DATADIR/data-bin/pairbc/ \
    --workers $W --srcdict $DATADIR/drug_id.dict \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.label \
    --validpref $DATADIR/valid.label \
    --destdir $DATADIR/data-bin/label/ \
    --workers $W --srcdict $DATADIR/label.dict \
