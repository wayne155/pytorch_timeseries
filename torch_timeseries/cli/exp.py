import sys
import fire
from torch_timeseries.experiments import get_experiment_class
from torch_timeseries.leaderboard import leaderboard


def _run_experiment():
    if '--model' not in sys.argv or '--task' not in sys.argv:
        print("Usage: --model MODEL --task TASK [hyperparams...]")
        print("  Regular tasks:   Forecast | Imputation | AnomalyDetection | UEAClassification | Generation")
        print("  Irregular tasks: IrregularClassification | IrregularInterpolation | IrregularForecast")
        print("  Models: DLinear | NLinear | Informer | Autoformer | FEDformer | PatchTST | iTransformer")
        print("          TSMixer | Crossformer | SCINet | TimesNet | CATS | FITS | FreTS")
        print("          GRUD | mTAN | LatentODE | NeuralCDE | Raindrop")
        print("  Generation models: TimeGAN | CSDI | DiffusionTS | TimeDiff | NsDiff | TMDM")
        print("Or: pytexp leaderboard --results_dir ./results --entries_dir leaderboard/entries")
        return

    model_index = sys.argv.index('--model') + 1
    task_index = sys.argv.index('--task') + 1

    if model_index >= len(sys.argv) or task_index >= len(sys.argv):
        print("No model or task specified after --model or --task")
        return

    model_name = sys.argv[model_index]
    task_name = sys.argv[task_index]

    # Pop the 4 argv entries (--model X --task Y) from largest index to
    # smallest so earlier pops don't shift later indices.
    for idx in sorted({model_index, model_index - 1, task_index, task_index - 1}, reverse=True):
        sys.argv.pop(idx)

    exp_class = get_experiment_class(model_name, task_name)

    fire.Fire(exp_class)


def compare(save_dir: str = "./results", task: str = None, dataset: str = None):
    """Print a comparison table from results saved in save_dir.

    Examples::

        pytexp compare --save_dir ./results --task Forecast
    """
    from torch_timeseries.experiment import Experiment
    Experiment.compare(save_dir=save_dir, task=task, dataset=dataset)


def exp():
    if len(sys.argv) > 1 and sys.argv[1] == "leaderboard":
        sys.argv.pop(1)
        fire.Fire(leaderboard)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        sys.argv.pop(1)
        fire.Fire(compare)
        return

    _run_experiment()


if __name__ == '__main__':
    exp()
