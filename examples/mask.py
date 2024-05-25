from torch_timeseries.dataset import ETTh1
from torch_timeseries.dataloader import StandardScaler, SlidingWindow, SlidingWindowTS, MaskTS
from torch_timeseries.model import DLinear
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
dataset = ETTh1('./data')
scaler = StandardScaler()
dataloader = MaskTS(dataset, 
                        window=96,
                        batch_size=32, 
                        train_ratio=0.7, 
                        val_ratio=0.2, 
                        scaler=scaler,
                        mask_rate=0.5
                        )


model = DLinear(dataloader.window, dataloader.window, dataset.num_features, individual= False)

optimizer = Adam(model.parameters())
loss_function = MSELoss()


# train
model.train()
for masked_scaled_x, scaled_x, x , mask, x_date_enc in dataloader.train_loader:
    optimizer.zero_grad()

    masked_scaled_x = masked_scaled_x.float()
    scaled_x = scaled_x.float()
    scaled_pred = model(masked_scaled_x) 

    loss = loss_function(scaled_pred[mask == 0], scaled_x[mask == 0])
    loss.backward()
    optimizer.step()
    print("train:", loss)
# val
model.eval()
for masked_scaled_x, scaled_x, x , mask, x_date_enc in dataloader.val_loader:

    masked_scaled_x = masked_scaled_x.float()
    scaled_x = scaled_x.float()
    scaled_pred = model(masked_scaled_x) 

    loss = loss_function(scaled_pred[mask == 0], scaled_x[mask == 0])
    print("val:", loss)
    

# test
model.eval()
for masked_scaled_x, scaled_x, x , mask, x_date_enc in dataloader.test_loader:
    masked_scaled_x = masked_scaled_x.float()
    scaled_x = scaled_x.float()
    scaled_pred = model(masked_scaled_x) 
    loss = loss_function(scaled_pred[mask == 0], scaled_x[mask == 0])
    print("test:", loss)
