from .DLinear import DLinear
from .NLinear import NLinear
from .Autoformer import Autoformer
from .FEDformer import FEDformer
from .Informer import Informer
from .PatchTST import PatchTST
from .iTransformer import iTransformer
from .TSMixer import TSMixer
from .Crossformer import Crossformer
from .SCINet import SCINet
from .TimesNet import TimesNet
from .CATS import CATS
from .FITS import FITS
from .FreTS import FreTS
from .TimeGAN import TimeGAN
from .CSDI import CSDI
from .DiffusionTS import DiffusionTS
from .TimeDiff import TimeDiff
from .NsDiff import NsDiff
from .TMDM import TMDM
from .diffusion_utils import GaussianDiffusion, make_beta_schedule, sinusoidal_embedding

forecasting_models = [
    "DLinear", "NLinear",
    "Informer", "Autoformer", "FEDformer",
    "PatchTST", "iTransformer",
    "TSMixer", "Crossformer", "SCINet", "TimesNet",
    "CATS", "FITS", "FreTS",
]

generation_models = [
    "TimeGAN", "CSDI", "DiffusionTS", "TimeDiff", "NsDiff", "TMDM",
]
