import subprocess


def test_forecast():
    result = subprocess.run(["python", "./torch_timeseries/cli/exp.py", "--model", "Autoformer", "--task", "Forecast", "--dataset_type", "ExchangeRate", "run", "3"], capture_output=True, text=True)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    print("Return code:", result.returncode)
# python ./torch_timeseries/cli/exp.py --model Autoformer --task Forecast --dataset_type ExchangeRate run 3
# python ./torch_timeseries/cli/exp.py --model Autoformer --task Imputation --dataset_type ExchangeRate run 3
# python ./torch_timeseries/cli/exp.py --model Autoformer --task AnomalyDetection --dataset_type MSL run 3
# python ./torch_timeseries/cli/exp.py --model Autoformer --task UEAClassification --dataset_type EthanolConcentration run 3


# python ./torch_timeseries/cli/exp.py --model FEDformer --task Forecast --dataset_type ExchangeRate run 3
# python ./torch_timeseries/cli/exp.py --model FEDformer --task Imputation --dataset_type ExchangeRate run 3
# python ./torch_timeseries/cli/exp.py --model FEDformer --task AnomalyDetection --dataset_type MSL run 3
# python ./torch_timeseries/cli/exp.py --model FEDformer --task UEAClassification --dataset_type EthanolConcentration run 3



# python ./torch_timeseries/cli/exp.py --model Informer --task Forecast --dataset_type ExchangeRate run 3
# python ./torch_timeseries/cli/exp.py --model Informer --task Imputation --dataset_type ExchangeRate run 3
# python ./torch_timeseries/cli/exp.py --model Informer --task AnomalyDetection --dataset_type MSL run 3
# python ./torch_timeseries/cli/exp.py --model Informer --task UEAClassification --dataset_type EthanolConcentration run 3



# python ./torch_timeseries/cli/exp.py --model PatchTST --task Forecast  --dataset_type ExchangeRate config_wandb --project=MARS --name=PatchTST run 3
# python ./torch_timeseries/cli/exp.py --model PatchTST --task Imputation --dataset_type ExchangeRate run 3
# python ./torch_timeseries/cli/exp.py --model PatchTST --task AnomalyDetection --dataset_type MSL run 3
# python ./torch_timeseries/cli/exp.py --model PatchTST --task UEAClassification --dataset_type EthanolConcentration run 3


# python ./torch_timeseries/cli/exp.py --model iTransformer --task Forecast  --dataset_type ExchangeRate config_wandb --project=MARS --name=PatchTST run 3
# python ./torch_timeseries/cli/exp.py --model iTransformer --task Imputation --dataset_type ExchangeRate run 3
# python ./torch_timeseries/cli/exp.py --model iTransformer --task AnomalyDetection --dataset_type MSL run 3
# python ./torch_timeseries/cli/exp.py --model iTransformer --task UEAClassification --dataset_type EthanolConcentration run 3