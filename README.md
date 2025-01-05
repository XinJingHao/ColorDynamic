<p align="center">
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/case1.gif" width="250" />
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/case_cd.gif" width="250" />
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/case2.gif" width="250" />
</p>



<div align=center>
  <img src="https://img.shields.io/badge/Python-blue" />
  <img src="https://img.shields.io/badge/DRL-blueviolet" />
  <img src="https://img.shields.io/badge/MobileRobot-ff69b4" />
</div>


## Dependency

```bash
numpy>=1.24.4
scipy>=1.10.1
pygame>=2.5.2
torch>=2.2.0 # GPU version is recommended
torchaudio>=2.2.0
torchvision>=0.17.0
```
## Quick Start
### Play with keyboard:

```bash
python play_with_keyboard.py
```

### Play with trained model:

```bash
python play_with_mouse.py
```
Then, use your mouse to set target points:
<p align="left">
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/play.gif" width="250" />
</p>


### Train your own model:

```bash
python train_ColorDynamic.py
```
During training, the model will be saved in ```model``` at a fixed frequency.

After training, you can use the following command to select the best model, where the evaluation results will be saved in ```Evaluation_result``` and ```runs```.
```bash
python evaluate_and_plot.py
```

## Navigation with ColorDynamic
Please refer to the [OPCD](https://github.com/XinJingHao/OPCD-Navigation) Navigation System.

<br/>

## Documentations:
- Sparrow-V2
- ColorDynamic
- [Color](https://www.sciencedirect.com/science/article/abs/pii/S0952197624018840)

<br/>

## Citing this Project

To cite this repository in publications:

```bibtex
@article{OPCD2025,
title = {OkayPlan and ColorDynamic},
journal = {XXX},
volume = {XXX},
pages = {XXX},
year = {2025},
issn = {XXXX-XXXX},
author = {Jinghao Xin},
}
```


