from torch_timeseries.dataset import UEA, M4, EthanolConcentration, FaceDetection, Handwriting, Heartbeat, JapaneseVowels, PEMS_SF, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits, UWaveGestureLibrary
from torch_timeseries.dataloader import StandardScaler, SlidingWindow, SlidingWindowTS
from torch_timeseries.model import DLinear
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam

dataset = UEA('EthanolConcentration')

dataset = EthanolConcentration('./data')
dataset = FaceDetection('./data')
dataset = Handwriting('./data')
dataset = Heartbeat('./data')
dataset = JapaneseVowels('./data')
dataset = PEMS_SF('./data')
dataset = SelfRegulationSCP1('./data')
dataset = SelfRegulationSCP2('./data')
dataset = SpokenArabicDigits('./data')
dataset = UWaveGestureLibrary('./data')

