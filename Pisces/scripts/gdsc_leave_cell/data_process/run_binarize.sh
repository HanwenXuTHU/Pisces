
# binarize dataset
DATAFOLD=fold$1
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/leave_cell/$DATAFOLD/

python fairseq_cli/preprocess.py \
    -s 'a' -t 'b' \
    --trainpref $DATADIR/train \
    --validpref $DATADIR/valid \
    --testpref $DATADIR/test \
    --destdir  $DATADIR/data-bin/ \
    --workers 30 --srcdict molecule/dict.txt \
    --joined-dictionary \
    --molecule \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.cell \
    --validpref $DATADIR/valid.cell \
    --testpref $DATADIR/test.cell \
    --destdir $DATADIR/data-bin/cell/ \
    --workers 30 --srcdict $DATADIR/cell.dict \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.pair \
    --validpref $DATADIR/valid.pair \
    --testpref $DATADIR/test.pair \
    --destdir $DATADIR/data-bin/pair/ \
    --workers 30 --srcdict $DATADIR/drug_id.dict \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.label \
    --validpref $DATADIR/valid.label \
    --testpref $DATADIR/test.label \
    --destdir $DATADIR/data-bin/label/ \
    --workers 30 --srcdict $DATADIR/label.dict \
