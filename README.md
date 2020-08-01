# SMAPGAN

Pytorch code for the [paper](https://arxiv.org/pdf/2001.07712) "SMAPGAN: Generative Adversarial Network-Based Semi-Supervised Styled Map Tiles Generation Method" by Xu Chen, Songqiang Chen, Tian Xu, Bangguo Yin, Jian Peng, Xiaoming Mei and Haifeng Li.

This project contains the implements of CycleGAN, Pix2pix, SMAPGAN and its ablation versions.

## Prerequisites

- Python 3
- PyTorch 0.4+ (check requirements.txt for more detail)
- NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo:

```bash
git clone https://github.com/imcsq/SMAPGAN
cd SMAPGAN
```

- Install [PyTorch](http://pytorch.org) and other dependencies.
  - For pip users, please type the command `pip install -r requirements.txt`.

### Train/test SMAPGAN

- Prepare and divide the related datasets. Original datasets could be found at: http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz, https://geods.geography.wisc.edu/archives/1192.

- Train a model:

```bash
python train.py --dataroot ./datasets/maps --name maps_smapgan --model smapgan
```

To see more intermediate results, check out `./checkpoints/maps_smapgan/web/index.html`.

- Test the model:

```bash
python test.py --dataroot ./datasets/maps --name maps_smapgan --model smapgan
```

- The test results will be saved to a html file here: `./results/maps_smapgn/latest_test/index.html`.

### Apply a pre-trained model

- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`. 

- Then generate the results using

```bash
python test.py --dataroot datasets/maps/testA --name maps_pretrained --model test --no_dropout
```

- The option `--model test` is used for generating results of SMAPGAN only for one side. This option will automatically set `--dataset_mode single`, which only loads the images from one set. On the contrary, using `--model smapgan` requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.
- For your own experiments, you might want to specify `--netG`, `--norm`, `--no_dropout` to match the generator architecture of the trained model.



## Acknowledgments

Our code is inspired by [pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
