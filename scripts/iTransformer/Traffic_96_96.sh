export PYTHONPATH=/notebooks/pytorchtimseries
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./torch_timeseries/experiments/iTransformer.py \
   --dataset_type="Traffic" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --e_layers=2 \
   --lr=0.0001 \
   --d_ff=2048 \
   --pred_len=96 \
   --windows=96 \
   --l2_weight_decay=0 \
   --epochs=50 \
   --patience=10 \
   runs --seeds='[1, 2, 3, 4, 5]'
