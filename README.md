## CircleGAN: Generative Adversarial Learning across Spherical Circles

This repository is the official implementation of CircleGAN. (https://arxiv.org/pdf/2011.12486.pdf)

Most code is adapted from [AM-GAN](https://github.com/ZhimingZhou/AM-GANs-refactored)

## Requirements

You can set up the required environments according to build_env.sh.

Or you can use docker with build_and_run.sh

## Training

* For training unconditional GANs, use the following command at directory /code:
```bash
python circlegan.py --sDataSet [stl10 or cifar10]
```

* For training conditional GANs, use the following command at directory /code:
```bash
python circlegan_cond.py --sDataSet [cifar10/cifar100/tinyimagenet]
```

- The datasets will be automatically downloaded.
- IS and FID will be evaluated at every 10% of total iterations.

## Citation

If you use this code or ideas for your research, please cite our paper.

```bib
@article{shim2020CircleGAN,
  title   = {{CircleGAN: Generative Adversarial Learning across Spherical Circles}},
  author  = {Woohyeon Shim and Minsu Cho},
  journal = {Conference on Neural Information Processing Systems (NeurIPS)},
  year    = {2020}
}
```
