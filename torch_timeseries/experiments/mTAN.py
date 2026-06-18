from dataclasses import dataclass

from ..model.irregular.mtan import mTAN
from .irregular_classification import IrregularClassificationExp
from .irregular_interpolation import IrregularInterpolationExp
from .irregular_forecast import IrregularForecastExp


@dataclass
class mTANParameters:
    hidden_size: int = 64
    num_ref_points: int = 16
    num_heads: int = 2
    mtan_dropout: float = 0.1


@dataclass
class mTANIrregularClassification(IrregularClassificationExp, mTANParameters):
    model_type: str = "mTAN"

    def _init_model(self) -> None:
        self.model = mTAN(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_classes,
            num_ref_points=self.num_ref_points,
            num_heads=self.num_heads,
            dropout=self.mtan_dropout,
        ).to(self.device)


@dataclass
class mTANIrregularInterpolation(IrregularInterpolationExp, mTANParameters):
    model_type: str = "mTAN"

    def _init_model(self) -> None:
        self.model = mTAN(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_features,
            num_ref_points=self.num_ref_points,
            num_heads=self.num_heads,
            dropout=self.mtan_dropout,
        ).to(self.device)


@dataclass
class mTANIrregularForecast(IrregularForecastExp, mTANParameters):
    model_type: str = "mTAN"

    def _init_model(self) -> None:
        self.model = mTAN(
            input_size=self.dm.num_features,
            hidden_size=self.hidden_size,
            output_size=self.dm.num_features,
            num_ref_points=self.num_ref_points,
            num_heads=self.num_heads,
            dropout=self.mtan_dropout,
        ).to(self.device)
