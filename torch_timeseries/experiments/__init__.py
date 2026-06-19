import os
import glob

from .forecast import ForecastExp
from .prob_forecast import ProbForecastExp
from .uea_classification import UEAClassificationExp
from .anomaly_detection import AnomalyDetectionExp
from .imputation import ImputationExp
import importlib

from .DLinear import (
    DLinearAnomalyDetection,
    DLinearForecast,
    DLinearImputation,
    DLinearUEAClassification,
)

from .NLinear import (
    NLinearAnomalyDetection,
    NLinearForecast,
    NLinearImputation,
    NLinearUEAClassification,
)

from .Autoformer import (
    AutoformerAnomalyDetection,
    AutoformerForecast,
    AutoformerImputation,
    AutoformerUEAClassification,
)

from .FEDformer import (
    FEDformerAnomalyDetection,
    FEDformerForecast,
    FEDformerImputation,
    FEDformerUEAClassification,
)

from .Informer import (
    InformerAnomalyDetection,
    InformerForecast,
    InformerImputation,
    InformerUEAClassification,
)

from .PatchTST import (
    PatchTSTAnomalyDetection,
    PatchTSTForecast,
    PatchTSTImputation,
    PatchTSTUEAClassification,
)

from .FITS import (
    FITSForecast,
    FITSUEAClassification,
    FITSAnomalyDetection,
    FITSImputation,
)

from .CATS import CATSForecast

from .TCNForecaster import (
    TCNForecast,
    TCNUEAClassification,
    TCNAnomalyDetection,
    TCNImputation,
)

from .PatchMixer import (
    PatchMixerForecast,
    PatchMixerUEAClassification,
    PatchMixerAnomalyDetection,
    PatchMixerImputation,
)

from .RNNForecaster import (
    RNNForecast,
    RNNUEAClassification,
    RNNAnomalyDetection,
    RNNImputation,
)

from .VanillaTransformer import (
    VanillaTransformerForecast,
    VanillaTransformerUEAClassification,
    VanillaTransformerAnomalyDetection,
    VanillaTransformerImputation,
)

model_list = ['iTransformer','TSMixer','TimesNet', 'SCINet', 'Crossformer', 'FITS', 'FreTS', 'SegRNN', 'TimeMixer', 'TiDE', 'NHiTS']

class_suffixes = ['AnomalyDetection', 'Forecast', 'Imputation', 'UEAClassification']
for model_prefix in model_list:
    for suffix in class_suffixes:
        module_name = f"{model_prefix}{suffix}"
        try:
            module = importlib.import_module(f".{model_prefix}", package=__name__)
            
            model_class = getattr(module, module_name, None)
            
            if model_class:
                globals()[module_name] = model_class
        except ModuleNotFoundError:
            print(f"Module {module_name} not found.")

from .irregular_classification import IrregularClassificationExp
from .irregular_interpolation import IrregularInterpolationExp
from .irregular_forecast import IrregularForecastExp
from .GRUD import (
    GRUDIrregularClassification,
    GRUDIrregularInterpolation,
    GRUDIrregularForecast,
)
from .mTAN import (
    mTANIrregularClassification,
    mTANIrregularInterpolation,
    mTANIrregularForecast,
)
from .LatentODE import (
    LatentODEIrregularClassification,
    LatentODEIrregularInterpolation,
    LatentODEIrregularForecast,
)
from .NeuralCDE import NeuralCDEIrregularClassification
from .Raindrop import RaindropIrregularClassification

from .MCDropoutForecaster import MCDropoutForecast
from .GaussianForecaster import GaussianForecast
from .StudentTForecaster import StudentTForecast
from .QuantileForecaster import QuantileForecast
from .EnsembleForecaster import EnsembleForecast
from .NormalizingFlowForecaster import NormalizingFlowForecast
from .NBEATS import (
    NBEATSForecast,
    NBEATSUEAClassification,
    NBEATSAnomalyDetection,
    NBEATSImputation,
)
from .SparseTSF import (
    SparseTSFForecast,
    SparseTSFUEAClassification,
    SparseTSFAnomalyDetection,
    SparseTSFImputation,
)
from .SOFTS import (
    SOFTSForecast,
    SOFTSUEAClassification,
    SOFTSAnomalyDetection,
    SOFTSImputation,
)
from .Koopa import (
    KoopaForecast,
    KoopaUEAClassification,
    KoopaAnomalyDetection,
    KoopaImputation,
)
from .LightTS import (
    LightTSForecast,
    LightTSUEAClassification,
    LightTSAnomalyDetection,
    LightTSImputation,
)
from .CycleNet import (
    CycleNetForecast,
    CycleNetUEAClassification,
    CycleNetAnomalyDetection,
    CycleNetImputation,
)
from .WaveNet import (
    WaveNetForecast,
    WaveNetUEAClassification,
    WaveNetAnomalyDetection,
    WaveNetImputation,
)
from .ETSformer import (
    ETSformerForecast,
    ETSformerUEAClassification,
    ETSformerAnomalyDetection,
    ETSformerImputation,
)
from .NSTransformer import (
    NSTransformerForecast,
    NSTransformerUEAClassification,
    NSTransformerAnomalyDetection,
    NSTransformerImputation,
)
from .MICN import (
    MICNForecast,
    MICNUEAClassification,
    MICNAnomalyDetection,
    MICNImputation,
)
from .TFT import (
    TFTForecast,
    TFTUEAClassification,
    TFTAnomalyDetection,
    TFTImputation,
)
from .FiLM import (
    FiLMForecast,
    FiLMUEAClassification,
    FiLMAnomalyDetection,
    FiLMImputation,
)
from .DishTS import (
    DishTSForecast,
    DishTSUEAClassification,
    DishTSAnomalyDetection,
    DishTSImputation,
)
from .MambaForecaster import (
    MambaForecasterForecast,
    MambaForecasterUEAClassification,
    MambaForecasterAnomalyDetection,
    MambaForecasterImputation,
)
from .ModernTCN import (
    ModernTCNForecast,
    ModernTCNUEAClassification,
    ModernTCNAnomalyDetection,
    ModernTCNImputation,
)
from .RLinear import (
    RLinearForecast,
    RLinearUEAClassification,
    RLinearAnomalyDetection,
    RLinearImputation,
)
from .FilterNet import (
    FilterNetForecast,
    FilterNetUEAClassification,
    FilterNetAnomalyDetection,
    FilterNetImputation,
)
from .CARD import (
    CARDForecast,
    CARDUEAClassification,
    CARDAnomalyDetection,
    CARDImputation,
)
from .Pathformer import (
    PathformerForecast,
    PathformerUEAClassification,
    PathformerAnomalyDetection,
    PathformerImputation,
)
from .SMamba import (
    SMambaForecast,
    SMambaUEAClassification,
    SMambaAnomalyDetection,
    SMambaImputation,
)
from .iMamba import (
    iMambaForecast,
    iMambaUEAClassification,
    iMambaAnomalyDetection,
    iMambaImputation,
)
from .Basisformer import (
    BasisformerForecast,
    BasisformerUEAClassification,
    BasisformerAnomalyDetection,
    BasisformerImputation,
)
from .GCNForecaster import (
    GCNForecasterForecast,
    GCNForecasterUEAClassification,
    GCNForecasterAnomalyDetection,
    GCNForecasterImputation,
)
from .MoEForecaster import (
    MoEForecasterForecast,
    MoEForecasterUEAClassification,
    MoEForecasterAnomalyDetection,
    MoEForecasterImputation,
)
from .RetForecaster import (
    RetForecasterForecast,
    RetForecasterUEAClassification,
    RetForecasterAnomalyDetection,
    RetForecasterImputation,
)
from .HarmonicForecaster import (
    HarmonicForecasterForecast,
    HarmonicForecasterUEAClassification,
    HarmonicForecasterAnomalyDetection,
    HarmonicForecasterImputation,
)
from .HDMixer import (
    HDMixerForecast,
    HDMixerUEAClassification,
    HDMixerAnomalyDetection,
    HDMixerImputation,
)
from .KANForecaster import (
    KANForecasterForecast,
    KANForecasterUEAClassification,
    KANForecasterAnomalyDetection,
    KANForecasterImputation,
)
from .TSReservoir import (
    TSReservoirForecast,
    TSReservoirUEAClassification,
    TSReservoirAnomalyDetection,
    TSReservoirImputation,
)
from .WaveletForecaster import (
    WaveletForecasterForecast,
    WaveletForecasterUEAClassification,
    WaveletForecasterAnomalyDetection,
    WaveletForecasterImputation,
)
from .LinearAttentionForecaster import (
    LinearAttentionForecasterForecast,
    LinearAttentionForecasterUEAClassification,
    LinearAttentionForecasterAnomalyDetection,
    LinearAttentionForecasterImputation,
)
from .DualDecompForecaster import (
    DualDecompForecasterForecast,
    DualDecompForecasterUEAClassification,
    DualDecompForecasterAnomalyDetection,
    DualDecompForecasterImputation,
)
from .HyperForecaster import (
    HyperForecasterForecast,
    HyperForecasterUEAClassification,
    HyperForecasterAnomalyDetection,
    HyperForecasterImputation,
)
from .GATForecaster import (
    GATForecasterForecast,
    GATForecasterUEAClassification,
    GATForecasterAnomalyDetection,
    GATForecasterImputation,
)
from .BiLSTMForecaster import (
    BiLSTMForecasterForecast,
    BiLSTMForecasterUEAClassification,
    BiLSTMForecasterAnomalyDetection,
    BiLSTMForecasterImputation,
)
from .RandomFourierForecaster import (
    RandomFourierForecasterForecast,
    RandomFourierForecasterUEAClassification,
    RandomFourierForecasterAnomalyDetection,
    RandomFourierForecasterImputation,
)
from .SparseTransformerForecaster import (
    SparseTransformerForecasterForecast,
    SparseTransformerForecasterUEAClassification,
    SparseTransformerForecasterAnomalyDetection,
    SparseTransformerForecasterImputation,
)
from .FourierMixerForecaster import (
    FourierMixerForecasterForecast,
    FourierMixerForecasterUEAClassification,
    FourierMixerForecasterAnomalyDetection,
    FourierMixerForecasterImputation,
)
from .TemporalConvAttentionForecaster import (
    TemporalConvAttentionForecasterForecast,
    TemporalConvAttentionForecasterUEAClassification,
    TemporalConvAttentionForecasterAnomalyDetection,
    TemporalConvAttentionForecasterImputation,
)
from .AdaptiveSpectralForecaster import (
    AdaptiveSpectralForecasterForecast,
    AdaptiveSpectralForecasterUEAClassification,
    AdaptiveSpectralForecasterAnomalyDetection,
    AdaptiveSpectralForecasterImputation,
)
from .LRUForecaster import (
    LRUForecasterForecast,
    LRUForecasterUEAClassification,
    LRUForecasterAnomalyDetection,
    LRUForecasterImputation,
)
from .S4Forecaster import (
    S4ForecasterForecast,
    S4ForecasterUEAClassification,
    S4ForecasterAnomalyDetection,
    S4ForecasterImputation,
)
from .HyenaForecaster import (
    HyenaForecasterForecast,
    HyenaForecasterUEAClassification,
    HyenaForecasterAnomalyDetection,
    HyenaForecasterImputation,
)
from .PrototypicalForecaster import (
    PrototypicalForecasterForecast,
    PrototypicalForecasterUEAClassification,
    PrototypicalForecasterAnomalyDetection,
    PrototypicalForecasterImputation,
)
from .MultiscaleConvForecaster import (
    MultiscaleConvForecasterForecast,
    MultiscaleConvForecasterUEAClassification,
    MultiscaleConvForecasterAnomalyDetection,
    MultiscaleConvForecasterImputation,
)
from .NeuralBasisForecaster import (
    NeuralBasisForecasterForecast,
    NeuralBasisForecasterUEAClassification,
    NeuralBasisForecasterAnomalyDetection,
    NeuralBasisForecasterImputation,
)
from .SincNetForecaster import (
    SincNetForecasterForecast,
    SincNetForecasterUEAClassification,
    SincNetForecasterAnomalyDetection,
    SincNetForecasterImputation,
)
from .GatedMLPForecaster import (
    GatedMLPForecasterForecast,
    GatedMLPForecasterUEAClassification,
    GatedMLPForecasterAnomalyDetection,
    GatedMLPForecasterImputation,
)
from .ImplicitNeuralForecaster import (
    ImplicitNeuralForecasterForecast,
    ImplicitNeuralForecasterUEAClassification,
    ImplicitNeuralForecasterAnomalyDetection,
    ImplicitNeuralForecasterImputation,
)
from .LiquidNetForecaster import (
    LiquidNetForecasterForecast,
    LiquidNetForecasterUEAClassification,
    LiquidNetForecasterAnomalyDetection,
    LiquidNetForecasterImputation,
)
from .xLSTMForecaster import (
    xLSTMForecasterForecast,
    xLSTMForecasterUEAClassification,
    xLSTMForecasterAnomalyDetection,
    xLSTMForecasterImputation,
)
from .EchoStateForecaster import (
    EchoStateForecasterForecast,
    EchoStateForecasterUEAClassification,
    EchoStateForecasterAnomalyDetection,
    EchoStateForecasterImputation,
)
from .AFTForecaster import (
    AFTForecasterForecast,
    AFTForecasterUEAClassification,
    AFTForecasterAnomalyDetection,
    AFTForecasterImputation,
)
from .RWKVForecaster import (
    RWKVForecasterForecast,
    RWKVForecasterUEAClassification,
    RWKVForecasterAnomalyDetection,
    RWKVForecasterImputation,
)
from .SpikeForecaster import (
    SpikeForecasterForecast,
    SpikeForecasterUEAClassification,
    SpikeForecasterAnomalyDetection,
    SpikeForecasterImputation,
)
from .ConformerForecaster import (
    ConformerForecasterForecast,
    ConformerForecasterUEAClassification,
    ConformerForecasterAnomalyDetection,
    ConformerForecasterImputation,
)
from .MEGAForecaster import (
    MEGAForecasterForecast,
    MEGAForecasterUEAClassification,
    MEGAForecasterAnomalyDetection,
    MEGAForecasterImputation,
)
from .MinGRUForecaster import (
    MinGRUForecasterForecast,
    MinGRUForecasterUEAClassification,
    MinGRUForecasterAnomalyDetection,
    MinGRUForecasterImputation,
)
from .FastFormerForecaster import (
    FastFormerForecasterForecast,
    FastFormerForecasterUEAClassification,
    FastFormerForecasterAnomalyDetection,
    FastFormerForecasterImputation,
)
from .GLAForecaster import (
    GLAForecasterForecast,
    GLAForecasterUEAClassification,
    GLAForecasterAnomalyDetection,
    GLAForecasterImputation,
)
from .DiffTransformerForecaster import (
    DiffTransformerForecasterForecast,
    DiffTransformerForecasterUEAClassification,
    DiffTransformerForecasterAnomalyDetection,
    DiffTransformerForecasterImputation,
)
from .QRNNForecaster import (
    QRNNForecasterForecast,
    QRNNForecasterUEAClassification,
    QRNNForecasterAnomalyDetection,
    QRNNForecasterImputation,
)

from .generation import GenerationExp
from .TimeGAN import TimeGANGeneration
from .CSDI import CSDIGeneration
from .DiffusionTS import DiffusionTSGeneration
from .TimeDiff import TimeDiffGeneration
from .NsDiff import NsDiffGeneration
from .TMDM import TMDMGeneration

from .registry import build_experiment_registry, format_experiment_choices

EXPERIMENT_REGISTRY = build_experiment_registry(globals())


def get_experiment_class(model_name: str, task_name: str):
    try:
        return EXPERIMENT_REGISTRY[(model_name, task_name)]
    except KeyError as exc:
        choices = format_experiment_choices(EXPERIMENT_REGISTRY)
        raise NotImplementedError(
            f"Unknown experiment: {model_name}{task_name}. "
            f"Available experiments: {choices}"
        ) from exc


def list_experiments():
    return sorted(EXPERIMENT_REGISTRY)
