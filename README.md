**Please star this repository if you like it :)**

**Feel free to leave feedbacks and ask questions. We want to make the repository helpful for your research.**

**We will keep updating this repository for pretrained models and weights.**

# UKBiobank_deep_pretrain
Pretrained neural networks for UK Biobank brain MRI images. SFCN, 3D-ResNet etc.

Under construction. 

The models are trained, validated and benchmarked with **UK Biobank brain MRI images, 14,503-subject release**.

Model input shape: \[batch_size, 1, 160, 192, 160]

## Pretrained weights (no subject level information) 

| File | Model | No. training subjects | Test MAE (years) |Validation MAE (yrs) |Train MAE (yrs) | Val-Train MAE gap (yrs) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|./brain_age/run_20190719_00_epoch_best_mae.p| SFCN (SGD) |	12,949 |2.14±0.05 | 2.18±0.04 |	1.36±0.03 |	0.83±0.06 |

(As summarized in Table 1 in the [manuscript](https://doi.org/10.1101/2019.12.17.879346))

## Examples
Checkout the file [**examples.ipynb**](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/blob/master/examples.ipynb)
```python
model = SFCN()
model = torch.nn.DataParallel(model)
# This is to be modified with the path of saved weights
p_ = './run_20190719_00_epoch_best_mae.p'
model.load_state_dict(torch.load(p_))
```

## Other resources
* **UK Biobank preprocessing information**
https://www.fmrib.ox.ac.uk/ukbiobank/fbp/

## To cite
Accurate brain age prediction with lightweight deep neural networks
Han Peng, Weikang Gong, Christian F. Beckmann, Andrea Vedaldi, Stephen M Smith
*Medical Image Analysis* (2021); doi: https://doi.org/10.1016/j.media.2020.101871


