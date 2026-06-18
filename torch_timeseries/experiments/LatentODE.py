from dataclasses import dataclass

from .irregular_classification import IrregularClassificationExp
from .irregular_interpolation import IrregularInterpolationExp
from .irregular_forecast import IrregularForecastExp


@dataclass
class LatentODEParameters:
    latent_size: int = 16
    hidden_size_ode: int = 32
    ode_method: str = "dopri5"


@dataclass
class LatentODEIrregularClassification(IrregularClassificationExp, LatentODEParameters):
    model_type: str = "LatentODE"

    def _init_model(self) -> None:
        from ..model.irregular.latent_ode import LatentODE
        self.model = LatentODE(
            input_size=self.dm.num_features,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size_ode,
            output_size=self.dm.num_classes,
            ode_method=self.ode_method,
        ).to(self.device)


@dataclass
class LatentODEIrregularInterpolation(IrregularInterpolationExp, LatentODEParameters):
    model_type: str = "LatentODE"

    def _init_model(self) -> None:
        from ..model.irregular.latent_ode import LatentODE
        self.model = LatentODE(
            input_size=self.dm.num_features,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size_ode,
            output_size=self.dm.num_features,
            ode_method=self.ode_method,
        ).to(self.device)


@dataclass
class LatentODEIrregularForecast(IrregularForecastExp, LatentODEParameters):
    model_type: str = "LatentODE"

    def _init_model(self) -> None:
        from ..model.irregular.latent_ode import LatentODE
        self.model = LatentODE(
            input_size=self.dm.num_features,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size_ode,
            output_size=self.dm.num_features,
            ode_method=self.ode_method,
        ).to(self.device)
