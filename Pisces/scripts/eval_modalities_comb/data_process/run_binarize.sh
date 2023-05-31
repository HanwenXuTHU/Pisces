
# binarize dataset
# DATADIR=/data/linjc/dds/data/transductive_3fold/$DATAFOLD/
W=32

DATAFOLD=$1
DATAROOTDIR='/home/swang/xuhw/research-projects/data/Pisces/transductive/fold'
DATADIR=$DATAROOTDIR$DATAFOLD
python fairseq_cli/preprocess.py \
    -s 'a' -t 'b' \
    --trainpref $DATADIR/train \
    --validpref $DATADIR/valid \
    --testpref $DATADIR/test \
    --destdir  $DATADIR/data-bin/ \
    --srcdict molecule/dict.txt \
    --joined-dictionary \
    --molecule \


python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.cell \
    --validpref $DATADIR/valid.cell \
    --testpref $DATADIR/test.cell \
    --destdir $DATADIR/data-bin/cell/ \
    --workers $W --srcdict $DATADIR/cell.dict \


python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.pair \
    --validpref $DATADIR/valid.pair \
    --testpref $DATADIR/test.pair \
    --destdir $DATADIR/data-bin/pair/ \
    --workers $W --srcdict $DATADIR/drug_id.dict \


python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.label \
    --validpref $DATADIR/valid.label \
    --testpref $DATADIR/test.label \
    --destdir $DATADIR/data-bin/label/ \
    --workers $W --srcdict $DATADIR/label.dict \
