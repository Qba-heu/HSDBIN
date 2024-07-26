# Hyperspherical Structural-aware Distillation Enhanced Spatial-Spectral Bidirectional Interaction Network for Hyperspectral Image Classification

This repository contains the `PyTorch` implementation for the submission [HSDBIN](https://ieeexplore.ieee.org/document/10608166).

```
@ARTICLE{10608166,
  author={Qin, Boao and Feng, Shou and Zhao, Chunhui and Xi, Bobo and Li, Wei and Tao, Ran and Li, Yunsong},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Hyperspherical Structural-aware Distillation Enhanced Spatial-Spectral Bidirectional Interaction Network for Hyperspectral Image Classification}, 
  year={2024},
  doi={10.1109/TGRS.2024.3433025}}

```

* This implementation is in PyTorch+GPU.

* This repo is a modification on the [MAE](https://arxiv.org/abs/2111.06377). Installation and detailed parameters follow that repo.

* This repo is based on
`python==3.9`,
[`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), 
`numpy==1.23.4`, 
`CUDA==11.3`,
`torch==1.11.0`.

The default dataset folder is `./Datasets/`.
### Running
* Start a Visdom serve: `python -m visdom.server` to see the visualization on http://localhost:8097/.
* running with disjoint datasets
```
    'python main_t2l.py --patch_size 19 --centroid_path './Estimated_prototypes/10centers_192dim.pth' --source_HSI PaviaU --disjoint True --epoch 100 --runs 5'
    'python main_t2l.py --patch_size 13 --centroid_path './Estimated_prototypes/16centers_192dim.pth' --source_HSI houston2013 --disjoint True --epoch 100 --runs 5'
    'python main_t2l.py --patch_size 13 --centroid_path './Estimated_prototypes/19centers_192dim.pth' --source_HSI YC --disjoint True --epoch 100 --runs 5 --alpha 0.1'
```
* If there is a new dataset that you want to test, you can re-estimate hyperspherical protatypes by
`python Protype_estimate.py --num_centroids {class num} --space_dim {embedding dim}`.

### Shell

- [x] sh file: `sh run_bash.sh`
- [x] run subprocess by: `python run_dir.py`

