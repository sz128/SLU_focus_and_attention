#!/bin/bash

bin=/home/slhome/sz128/src/torch/install/bin/luajit

dropout=0.5
rnn_type=bilstm_lstm_focus
for lr in 0.0325 #0.0225 0.025 0.0275 0.0325 0.035 0.0375 0.0425 0.045 #0.008 0.01 0.02 0.03 0.04
do
  ./run_lstm.sh --bin $bin --rnn_type $rnn_type --dropout $dropout --me 100 --wl 0 --wr 0 --es 100 --proto "100" --mb 10 --lr $lr --bs 2 --di 1 --mb_test 10
done
