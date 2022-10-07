# Learning Diverse Tone Styles for Image Retouching

This is the official repository for our paper:
**Learning Diverse Tone Styles for Image Retouching**
> [http://arxiv.org/abs/2207.05430](http://arxiv.org/abs/2207.05430)


## Pre-train Models
Download the pretrained model from the following url and put them into ./experiments/pre_merge/models/
- [BaiduNetDisk](https://pan.baidu.com/s/1PN8pnuhQL32pb5USxLgRcQ?pwd=l93a [l93a])

## Quick Training
```
cd codes

CUDA_VISIBLE_DEVICES=0 python trainmerge.py -opt ./confs/pre_merge.yml
```
## Testing
```
Change the dataset dir in testfiveksingle.py

CUDA_VISIBLE_DEVICES=0 python testfiveksingle.py ./confs/pre_merge.yml
```

## Acknowledgement
This project is built based on the excellent [SRFlow] (https://github.com/andreas128/SRFlow)

## Citation

```
@article{wang2022learning,
  title={Learning Diverse Tone Styles for Image Retouching},
  author={Wang, Haolin and Zhang, Jiawei and Liu, Ming and Wu, Xiaohe and Zuo, Wangmeng},
  journal={arXiv preprint arXiv:2207.05430},
  year={2022}
}
```
