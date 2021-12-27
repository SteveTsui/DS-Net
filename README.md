# An unofficial implementation of  [Dual-stream Network for Visual Recognition](https://arxiv.org/abs/2105.14734) 

## Prerequisites 

* torch==1.7.0

* torchvision==0.8.1

* timm==0.3.2

## Performance

|Model|Params|Top-1 acc|Download|
| :-------: | :-------: | :-------: | :-------: |
|DS-Net-T|10.5|79.0|[model](https://drive.google.com/file/d/1LQN95zSfmROSh7AVDAVTWCqH3EHZcATv/view?usp=sharing)|
|DS-Net-S|23|82.3|[model](https://drive.google.com/file/d/1aqMAimClkkFBHX_VzJ8121XCdbYPz777/view?usp=sharing)|

## Evaluation
```
sh ./run_eval.sh
```

## Training
```
sh ./run_train.sh
```

# DS-Net
