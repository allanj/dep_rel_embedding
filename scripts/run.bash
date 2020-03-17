#!/bin/bash

datasets=(catalan spanish)
num_epochs_all=(500 500)
devices=(cuda:1 cuda:1)   ##cpu, cuda:0, cuda:1
dep_model=dggcn  ## for pretrainning
batch_size=32

for (( d=0; d<${#datasets[@]}; d++  )) do
    dataset=${datasets[$d]}
    device=${devices[$d]}
    num_epochs=${num_epochs_all[$d]}
    model_folder=${datasets[$d]}_${num_epochs}
    logfile=logs/pretrained_${dataset}_${dep_model}.log
    python3.6 label_main.py --dataset ${dataset}  --num_epochs ${num_epochs} --device ${device}   \
        --dep_model ${dep_model} --batch_size ${batch_size} --model_folder ${model_folder} > ${logfile} 2>&1
done



