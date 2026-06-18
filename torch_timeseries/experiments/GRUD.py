from dataclasses import dataclass

from ..model.irregular.grud import GRUD
from .irregular_classification import IrregularClassificationExp
from .irregular_interpolation import IrregularInterpolationExp
from .irregular_forecast import IrregularForecastExp


@dataclass
class GRUDParameters:
    hidden_size: int = 64
    grud_dropout: float = 0.0


@dataclass
class GRUDIrregularClassification(IrregularClassificationExp, GRUDParameters):
    model_type: str = "GRUD"

    def _init_model(self) -> None:
        self.model = GRUD(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_classes,
            dropout=self.grud_dropout,
        ).to(self.device)


@dataclass
class GRUDIrregularInterpolation(IrregularInterpolationExp, GRUDParameters):
    model_type: str = "GRUD"

    def _init_model(self) -> None:
        self.model = GRUD(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_features,   # seq2seq: output_size unused, input_size drives fc_seq2seq
            dropout=self.grud_dropout,
        ).to(self.device)


@dataclass
class GRUDIrregularForecast(IrregularForecastExp, GRUDParameters):
    model_type: str = "GRUD"

    def _init_model(self) -> None:
        self.model = GRUD(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_features,
            dropout=self.grud_dropout,
        ).to(self.device)
