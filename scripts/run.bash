#!/bin/bash

datasets=(ontonotes ontonotes)
num_epochs_all=(500 500)
devices=(cuda:0 cuda:1)   ##cpu, cuda:0, cuda:1
dep_model=dggcn  ## for pretrainning
batch_size=32
complete_trees=(0 1)

for (( d=0; d<${#datasets[@]}; d++  )) do
    dataset=${datasets[$d]}
    device=${devices[$d]}
    num_epochs=${num_epochs_all[$d]}
    complete_tree=${complete_trees[$d]}
    model_folder=${datasets[$d]}_${num_epochs}_levi_${complete_tree}
    logfile=logs/pretrained_${model_folder}.log
    python3.6 label_main.py --dataset ${dataset}  --num_epochs ${num_epochs} --device ${device} --complete_tree ${complete_tree}  \
        --dep_model ${dep_model} --batch_size ${batch_size} --model_folder ${model_folder} > ${logfile} 2>&1  &
done



