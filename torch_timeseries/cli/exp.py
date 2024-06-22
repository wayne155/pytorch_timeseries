import sys
import fire
from torch_timeseries.experiments import *
def exp():
    if '--model' in sys.argv and '--task' in sys.argv:
        model_index = sys.argv.index('--model') + 1
        task_index = sys.argv.index('--task') + 1
        
        if model_index < len(sys.argv) and task_index < len(sys.argv):
            model_name = sys.argv[model_index]
            task_name = sys.argv[task_index]
            
            sys.argv.pop(task_index)
            sys.argv.pop(task_index - 1)
            sys.argv.pop(model_index)
            sys.argv.pop(model_index - 1)
            
            try:
                model_exp = f"{model_name}{task_name}"
                exp_class = eval(model_exp)
            except NameError:
                raise NotImplementedError(f"Unknown experiment: {model_name}{task_name}")
            fire.Fire(exp_class)
        else:
            print("No model or task specified after --model or --task")
    else:
        print("Usage: --model [DLinear] --task [Forecast|Imputation|Classification|AnomalyDetection]")

if __name__ == '__main__':
    exp()
