import numpy as np
import torch
from torch_timeseries.dataset import SMD, SWaT, MSL, PSM, SMAP
from torch_timeseries.dataloader import StandardScaler, AnomalyLoader
from torch_timeseries.model import DLinear
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred
scaler = StandardScaler()


dataset = SWaT('./data')
# dataset = SMD('./data')
# dataset = PSM('./data')
# dataset = MSL('./data')
# dataset = SMAP('./data')


scaler = StandardScaler()

dataloader = AnomalyLoader(dataset, scaler, 168, True, 32, 0.8 ,3)

model = DLinear(dataloader.window, dataloader.window, dataset.num_features, individual= False)

optimizer = Adam(model.parameters())
loss_function = MSELoss()
anomaly_ratio = 0.25

train_loss = []
for scaled_x, x in dataloader.train_loader:
    optimizer.zero_grad()
    outputs = model(scaled_x)
    loss = loss_function(outputs, scaled_x)
    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())
    print(loss)



# def vali(_loader):
#     total_loss = []
#     model.eval()
#     with torch.no_grad():
#         for i, (batch_x, _) in enumerate(_loader):
#             outputs = model(batch_x)
#             pred = outputs.detach().cpu()
#             true = batch_x.detach().cpu()
#             loss = loss_function(pred, true)
#             total_loss.append(loss)
#     total_loss = np.average(total_loss)
#     return total_loss




print("val................................")
model.eval()
for scaled_x, x in dataloader.val_loader:
    outputs = model(scaled_x)
    loss = loss_function(outputs, scaled_x)
    print("loss", loss)
    
    
print("test................................")
def test():
    anomaly_criterion = torch.nn.MSELoss(reduce=False)
    attens_energy = []

    model.eval()

    # (1) stastic on the train set
    with torch.no_grad():
        for i, (scaled_x, _) in enumerate(dataloader.train_loader):
            outputs = model(scaled_x)
            score = torch.mean(anomaly_criterion(scaled_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)

    # (2) find the threshold
    attens_energy = []
    test_labels = []
    for i, (batch_x, _, batch_y) in enumerate(dataloader.test_loader):
        outputs = model(batch_x)
        score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
        score = score.detach().cpu().numpy()
        attens_energy.append(score)
        test_labels.append(batch_y)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
    print("Threshold :", threshold)

    # (3) evaluation on the test set
    pred = (test_energy > threshold).astype(int)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_labels = np.array(test_labels)
    gt = test_labels.astype(int)

    print("pred:   ", pred.shape)
    print("gt:     ", gt.shape)

    # (4) detection adjustment
    gt, pred = adjustment(gt, pred)

    pred = np.array(pred)
    gt = np.array(gt)
    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))
    return


test()

