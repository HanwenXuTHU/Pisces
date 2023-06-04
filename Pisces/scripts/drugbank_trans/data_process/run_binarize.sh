
# binarize dataset

DATADIR=/home/swang/xuhw/research-projects/data/Pisces/drugbank_trans/fold$1

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
    --trainpref $DATADIR/train.pair \
    --validpref $DATADIR/valid.pair \
    --testpref $DATADIR/test.pair \
    --destdir $DATADIR/data-bin/pair/ \
    --workers 30 --srcdict $DATADIR/drug_id.dict \

python fairseq_cli/preprocess.py \
    -s 'nega' -t 'negb' \
    --trainpref $DATADIR/train \
    --validpref $DATADIR/valid \
    --testpref $DATADIR/test \
    --destdir $DATADIR/data-bin/ \
    --workers 30 --srcdict molecule/dict.txt \
    --joined-dictionary \
    --molecule \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.label \
    --validpref $DATADIR/valid.label \
    --testpref $DATADIR/test.label \
    --destdir $DATADIR/data-bin/label/ \
    --workers 30 --srcdict $DATADIR/label.dict \

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.negpair \
    --validpref $DATADIR/valid.negpair \
    --testpref $DATADIR/test.negpair \
    --destdir $DATADIR/data-bin/negpair/ \
    --workers 30 --srcdict $DATADIR/drug_id.dict \