#!/bin/bash

bin=/home/slhome/sz128/src/torch/install/bin/luajit

dropout=0.5
for lr in 0.006 #0.008 0.01 0.015
do
  ./run_lstm.sh --bin $bin --rnn_type bilstm_lstm_attention --dropout $dropout --me 100 --wl 0 --wr 0 --mb 10 --lr $lr --bs 2 --di 1 --mb_test 10
done
