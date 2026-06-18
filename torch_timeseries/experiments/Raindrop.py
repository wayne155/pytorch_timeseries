from dataclasses import dataclass

from .irregular_classification import IrregularClassificationExp


@dataclass
class RaindropParameters:
    raindrop_hidden_size: int = 32
    raindrop_num_heads: int = 2
    raindrop_dropout: float = 0.1


@dataclass
class RaindropIrregularClassification(IrregularClassificationExp, RaindropParameters):
    model_type: str = "Raindrop"

    def _init_model(self) -> None:
        from ..model.irregular.raindrop import Raindrop
        self.model = Raindrop(
            input_size=self.dm.num_features,
            hidden_size=self.raindrop_hidden_size,
            output_size=self.dm.num_classes,
            num_nodes=self.dm.num_features,
            num_heads=self.raindrop_num_heads,
            dropout=self.raindrop_dropout,
        ).to(self.device)
