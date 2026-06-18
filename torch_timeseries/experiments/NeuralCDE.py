from dataclasses import dataclass

from .irregular_classification import IrregularClassificationExp


@dataclass
class NeuralCDEParameters:
    ncde_hidden_size: int = 32
    interpolation: str = "cubic"


@dataclass
class NeuralCDEIrregularClassification(IrregularClassificationExp, NeuralCDEParameters):
    model_type: str = "NeuralCDE"

    def _init_model(self) -> None:
        from ..model.irregular.neural_cde import NeuralCDE
        self.model = NeuralCDE(
            input_size=self.dm.num_features,
            hidden_size=self.ncde_hidden_size,
            output_size=self.dm.num_classes,
            interpolation=self.interpolation,
        ).to(self.device)
