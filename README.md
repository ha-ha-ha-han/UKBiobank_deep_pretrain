# UKBiobank_deep_pretrain
Pretrained neural networks for UK Biobank brain MRI images. SFCN, 3D-ResNet etc.

Under construction.

The models are trained, validated and benchmarked with **UK Biobank brain MRI images, 14,503-subject release**.

Pretrained weights for SFCN:
./brain_age/run_20190719_00_epoch_best_mae.p

Example:
```python
model = SFCN()
model = torch.nn.DataParallel(model)
# This is to be modified with the path of saved weights
p_ = './run_20190719_00_epoch_best_mae.p'
model.load_state_dict(torch.load(p_))
```

To cite:
Accurate brain age prediction with lightweight deep neural networks
Han Peng, Weikang Gong, Christian F. Beckmann, Andrea Vedaldi, Stephen M Smith
*bioRxiv 2019.12.17.879346*; doi: https://doi.org/10.1101/2019.12.17.879346

Model input shape: \[batch_size, 1, 160, 192, 160]
