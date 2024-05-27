.. vim: syntax=rst

quick start
======================

Here is an example to train DLinear model in a long-term forecast settings, see ../examples/quickstart.py  for more details

.. code-block:: python 
    :caption: quick start
    :linenos:

    from torch_timeseries.dataset import ETTh1
    from torch_timeseries.dataloader import StandardScaler, SlidingWindow, SlidingWindowTS
    from torch_timeseries.model import DLinear
    from torch.nn import MSELoss, L1Loss
    from torch.optim import Adam
    dataset = ETTh1('./data')
    scaler = StandardScaler()
    dataloader = SlidingWindowTS(dataset, 
        window=96,
        horizon=1,
        steps=336,
        batch_size=32, 
        train_ratio=0.7, 
        val_ratio=0.2, 
        scaler=scaler,
        )
    model = DLinear(dataloader.window, dataloader.steps, dataset.num_features, individual= True)

    optimizer = Adam(model.parameters())
    loss_function = MSELoss()


    # train
    model.train()
    for scaled_x, scaled_y, x, y, x_date_enc, y_date_enc in dataloader.train_loader:
        optimizer.zero_grad()
        
        scaled_x = scaled_x.float()
        scaled_y = scaled_y.float()
        scaled_pred_y = model(scaled_x) 
        
        loss = loss_function(scaled_pred_y, scaled_y)
        loss.backward()
        optimizer.step()
        print(loss)


