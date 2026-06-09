#!/bin/bash
# bash ./run_scripts/run_wandb.sh "PatchTST" "ExchangeRate" "96" "cuda:0" 336 "1 2 3 4 5"
declare -A dataset_to_window_map
models=($1)
datasets=($2)  
pred_lens=($3)      
device=$4
windows=$5
seeds=($6)
for model in "${models[@]}"
    do
    for pred_len in "${pred_lens[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            for seed in "${seeds[@]}"
            do
                echo $seed
                CUDA_DEVICE_ORDER=PCI_BUS_ID pytexp --model $model --task Forecast --lr=0.001 --experiment_label="seed_$seed" --dataset_type="$dataset"  --device="$device" --batch_size=32 --horizon=1 --pred_len="$pred_len" --windows=$windows --epochs=100 config_wandb --project=ForecastBase --name="$model-$windows-$pred_len"  run --seed=$seed
            done
        done
    done
done

echo "All runs completed."
