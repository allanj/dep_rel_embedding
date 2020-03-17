#!/bin/bash

datasets=(ontonotes ontonotes)
context_emb=none
num_epochs_all=(100 100)
devices=(cuda:1 cuda:1)   ##cpu, cuda:0, cuda:1
dep_model=dglstm  ## none, dglstm, dggcn means do not use head features
embs=(data/glove.6B.100d.txt data/glove.6B.100d.txt)
num_lstm_layer=1
inter_func=mlp
train_num=-1
pretrain_deps=(1 0)
freezes=(0 1)
dep_emb_size=200

for (( d=0; d<${#datasets[@]}; d++  )) do
    dataset=${datasets[$d]}
    emb=${embs[$d]}
    device=${devices[$d]}
    num_epochs=${num_epochs_all[$d]}
    pretrain_dep=${pretrain_deps[$d]}
    freeze=${freezes[$d]}
    model_folder=${dataset}_${dep_model}_pdep_${pretrain_dep}_freeze_${freeze}
    first_part=logs/hidden_${num_lstm_layer}_${dataset}_${train_num}_${dep_model}_asfeat_${context_emb}
    logfile=${first_part}_epoch_${num_epochs}_if_${inter_func}_pretrain_dep_${pretrain_dep}_freeze_${freeze}.log
    python3.6 main.py --context_emb ${context_emb}  --train_num ${train_num}\
      --dataset ${dataset}  --num_epochs ${num_epochs} --device ${device}  --num_lstm_layer ${num_lstm_layer} \
        --dep_model ${dep_model} --pretrain_dep ${pretrain_dep} --model_folder ${model_folder} \
       --embedding_file ${emb} --inter_func ${inter_func} --dep_emb_size ${dep_emb_size} > ${logfile} 2>&1

done


