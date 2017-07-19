#!/bin/bash

bin=/speechlab/users/sz128/src/torch_new_mh/install/bin/luajit
mb=10        # minibatch for training in parallel
lr=0.03      # learning rate, e.g. 0.0627142536696559
coefL2=0     # coefficient of L2 norm
max_norm=0   #
es=100       # word embedding size
es_bio=4      # bio embedding size
es_head=10    # head embedding size
es_tail=40    # tail embedding size
proto="100"       # hidden layer prototype, e.g. 100-200-300
wl=0         # left context window of the current word
wr=2         # right context window of the current word
me=50         # max epoch
bs=2          # beam size
rnn_type=lstm # network type
mb_test=10        # minibatch for testing in parallel
di=1         # -deviceId
rdseed=999
dropout=0

echo "$0 $@"
. ./utils/parse_options.sh || exit 1

main_lua=main.lua
expdir=exp/${rnn_type}/atis_mb${mb}_lr${lr}_window${wl}-${wr}_es${es}_proto${proto}_dropout${dropout}_coefL2${coefL2}_mn${max_norm}_me${me}

[ ! -d $expdir ] && mkdir -p $expdir

datadir=data/ms_rnn_atis/atis.fold3
cp $datadir/train.20 $expdir/train
cp $datadir/valid.20 $expdir/valid
cp $datadir/test.20 $expdir/test
vocab="-vocab $expdir/train -outlabel $datadir/idx2la -print_vocab $expdir/vocab"

$bin $main_lua -train $expdir/train -valid $expdir/valid -test $expdir/test \
  $vocab \
  -rnn_type ${rnn_type} \
  -print_model $expdir/slu.rnn \
  -max_epoch $me \
  -hidden_prototype $proto -emb_size $es \
  -word_win_left $wl -word_win_right $wr \
  -batch_size $mb -test_batch_size $mb_test \
  -alpha_decay 0.6 -alpha $lr -coefL2 $coefL2 -max_norm ${max_norm} -dropout $dropout \
  -init_weight 0.2 -random_seed $rdseed \
  -beam_size $bs \
  -deviceId $di \
  | tee $expdir/log.txt

