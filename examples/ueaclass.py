import numpy as np
from torch_timeseries.scaler import StandardScaler
from torch_timeseries.dataset import UEA
from torch_timeseries.dataloader import UEAClass
from torch_timeseries.model import DLinear
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import torch

uea = UEA('SpokenArabicDigits', './data')
scaler = StandardScaler()
dataloader = UEAClass(uea, scaler, 1700)

model = DLinear(1700, 48, uea.num_features, False, output_prob=uea.num_classes)
optimizer = Adam(model.parameters())
loss_func = CrossEntropyLoss()

def accuracy(preds, trues):
    probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
    predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    trues = trues.flatten().cpu().numpy()
    accuracy = np.mean(predictions == trues)
    return accuracy

print("training................................")
model.train()
for scaled_x, x, y, padding_masks in dataloader.train_loader:
    optimizer.zero_grad()
    
    probs = model(scaled_x)
    loss = loss_func(probs, y.long().squeeze(-1))
    
    loss.backward()
    optimizer.step()

    print(loss)
    print("acc", accuracy(probs, y))


print("val................................")
model.eval()
for scaled_x, x, y, padding_masks in dataloader.val_loader:
    probs = model(scaled_x)
    loss = loss_func(probs, y.long().squeeze(-1))
    print(loss)
    
    print("acc", accuracy(probs, y))
print("test................................")
model.eval()
for scaled_x, x, y, padding_masks in dataloader.test_loader:
    probs = model(scaled_x)
    loss = loss_func(probs, y.long().squeeze(-1))
    print(loss)
    print("acc", accuracy(probs, y))
