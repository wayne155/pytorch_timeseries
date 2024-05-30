import sys
from torch_timeseries.apps.DLinear import DLinearForecast, DLinearImputation, DLinearUEAClassification, DLinearAnomalyDetection


def main():
    import fire
    if '--task' in sys.argv:
        task_index = sys.argv.index('--task') + 1
        if task_index < len(sys.argv):
            task_name = sys.argv[task_index]
            sys.argv.pop(task_index)
            sys.argv.pop(task_index - 1)
            if task_name == 'Forecast':
                fire.Fire(DLinearForecast)
            elif task_name == 'Imputation':
                fire.Fire(DLinearImputation)
            elif task_name == 'UEAClassification':
                fire.Fire(DLinearUEAClassification)
            elif task_name == 'AnomalyDetection':
                fire.Fire(DLinearAnomalyDetection)
            else:
                print(f"Unknown task: {task_name}")
        else:
            print("No task specified after --task")
    else:
        print("Usage: DLinear.py --task [Forecast|Imputation|Classification|AnomalyDetection]")

if __name__ == '__main__':
    main()
