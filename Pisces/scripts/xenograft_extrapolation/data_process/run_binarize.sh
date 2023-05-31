
# binarize dataset
for i in {0..9}
do
DATAFOLD=fold$i
DATADIR=/home/swang/xuhw/research-projects/data/Pisces/xenograft_data_10fold/response_days_v2/$DATAFOLD

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
    --trainpref $DATADIR/train.model \
    --validpref $DATADIR/valid.model \
    --testpref $DATADIR/test.model \
    --destdir $DATADIR/data-bin/model/ \
    --workers 30 --srcdict $DATADIR/model.dict \

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

python fairseq_cli/preprocess.py \
    --only-source \
    --trainpref $DATADIR/train.t \
    --validpref $DATADIR/valid.t \
    --testpref $DATADIR/test.t \
    --destdir $DATADIR/data-bin/time/ \
    --workers 30 --srcdict $DATADIR/time.dict
done