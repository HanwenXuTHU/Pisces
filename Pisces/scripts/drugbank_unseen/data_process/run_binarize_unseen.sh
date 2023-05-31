
# binarize dataset

DATADIR=/home/swang/xuhw/research-projects/data/Pisces/drugbank_unseen/fold$1

python fairseq_cli/preprocess.py \
    -s 'a' -t 'b' \
    --trainpref $DATADIR/train \
    --validpref $DATADIR/valid \
    --destdir  $DATADIR/data-bin/ \
    --workers 30 --srcdict molecule/dict.txt \
    --joined-dictionary \
    --molecule \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.pair \
    --validpref $DATADIR/valid.pair \
    --destdir $DATADIR/data-bin/pair/ \
    --workers 30 --srcdict $DATADIR/drug_id.dict \

python fairseq_cli/preprocess.py \
    -s 'nega' -t 'negb' \
    --trainpref $DATADIR/train \
    --validpref $DATADIR/valid \
    --destdir $DATADIR/data-bin/ \
    --workers 30 --srcdict molecule/dict.txt \
    --joined-dictionary \
    --molecule \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.label \
    --validpref $DATADIR/valid.label \
    --destdir $DATADIR/data-bin/label/ \
    --workers 30 --srcdict $DATADIR/label.dict \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.negpair \
    --validpref $DATADIR/valid.negpair \
    --destdir $DATADIR/data-bin/negpair/ \
    --workers 30 --srcdict $DATADIR/drug_id.dict \