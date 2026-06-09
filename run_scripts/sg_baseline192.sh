#!/bin/bash

# bash ./run_scripts/sg_baseline.sh cuda:0 "TimesNet" "ETTh1 ETTh2 ETTm1 ETTm2 Traffic" 
# bash ./run_scripts/sg_baseline.sh cuda:3 "FITS FreTS PatchTST iTransformer" "Electricity" 
# bash ./run_scripts/sg_baseline192.sh cuda:2 "FITS FreTS PatchTST iTransformer" 

# 固定参数
task="Forecast"
runs="[1,2,3]"
windows="96"
pred_lens=(192)
# pred_lens=(168)

# 默认 device
device="cuda:0"

# 默认模型列表（可被命令行覆盖）
default_models=("DLinear")

# 默认数据集列表（按你要求更新）
default_datasets=(
    "ETTh1"
    "ETTh2"
    "ETTm1"
    "ETTm2"
    "Weather"
    "SolarEnergy"
    "Traffic"
    "Electricity"
    "ExchangeRate"
)

# default_datasets=(
#     "ETTh1"
#     "ETTm1"
#     "Weather"
#     "SolarEnergy"
#     "Traffic"
#     "Electricity"
# )
# 解析命令行参数
if [ $# -ge 1 ]; then
    device="$1"
fi
if [ $# -ge 2 ]; then
    read -ra model_list <<< "$2"
else
    model_list=("${default_models[@]}")
fi
if [ $# -ge 3 ]; then
    read -ra dataset_type_list <<< "$3"
else
    dataset_type_list=("${default_datasets[@]}")
fi

echo "Device: $device"
echo "Models: ${model_list[*]}"
echo "Datasets: ${dataset_type_list[*]}"
echo "Pred Lens: ${pred_lens[*]}"
echo "Windows: $windows"
echo "----------------------------------------"

# 遍历所有组合
for model in "${model_list[@]}"; do
    for dataset in "${dataset_type_list[@]}"; do
        for pred_len in "${pred_lens[@]}"; do
            echo "Running: model=$model, dataset=$dataset, pred_len=$pred_len, device=$device"
            pytexp \
                --model "$model" \
                --task "$task" \
                --columns=[0] \
                --dataset_type "$dataset" \
                --windows "$windows" \
                --pred_len "$pred_len" \
                --device "$device" \
                config_wandb ForecastBaseSG \
                runs "$runs"
        done
    done
done