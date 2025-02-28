<p align="center">
  <img src="https://github.com/XinJingHao/Images/blob/main/ColorDynamic/logo.png" width="300" /><br>
  <img src="https://img.shields.io/badge/Python-blue" />
  <img src="https://img.shields.io/badge/DRL-blueviolet" />
  <img src="https://img.shields.io/badge/MobileRobot-ff69b4" />
</p>

<br/>

<p align="center">
  <strong>Generalizable, Scalable, Real-time, End-to-end Local Planner</strong><br>
  <strong>for Unstructured & Dynamic Environments</strong>
</p>

<br/>

## Gallery
<p align="center">
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/case2.gif" width="250" />
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/case_cd.gif" width="250" />
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V3/N10.gif" width="250" />
</p>

<p align="center">
  <img src="https://github.com/XinJingHao/Images/blob/main/ColorDynamic/warehouse1.gif" width="250" />
  <img src="https://github.com/XinJingHao/Images/blob/main/ColorDynamic/warehouse2.gif" width="250" />
  <img src="https://github.com/XinJingHao/Images/blob/main/ColorDynamic/warehouse3.gif" width="250" />
</p>

<p align="center">
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/real1.gif" width="250" />
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/real2.gif" width="250" />
  <img src="https://github.com/XinJingHao/Images/blob/main/Sparrow_V2/real3.gif" width="250" />
</p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
The video can be found [here](https://www.youtube.com/watch?v=5NMHKDOdt54). 

## Dependency

```bash
numpy>=1.24.4
scipy>=1.10.1
pygame>=2.5.2
torch>=2.2.0 # GPU version is recommended
torchaudio>=2.2.0
torchvision>=0.17.0
tensorboard>=2.18.0
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
<p align="left">
  <img src="https://github.com/XinJingHao/Images/blob/main/ColorDynamic/warehouse.gif" width="500" />
</p>

Please refer to the [OPCD](https://github.com/XinJingHao/OPCD-Navigation) Navigation System.

<br/>

## Documentations:
- [Sparrow-V2](https://github.com/XinJingHao/Sparrow-V2)
- [Sparrow-V3](https://github.com/XinJingHao/Sparrow-V3)
- [ColorDynamic](https://github.com/XinJingHao/ColorDynamic)
- [Color](https://www.sciencedirect.com/science/article/abs/pii/S0952197624018840)
- [OPCD](https://github.com/XinJingHao/OPCD-Navigation)

<br/>

## Citing this Project

To cite this repository in publications:

```bibtex
@article{ColorDynamic,
  title={ColorDynamic: Generalizable, Scalable, Real-time, End-to-end Local Planner for Unstructured and Dynamic Environments},
  author={Jinghao Xin, Zhichao Liang, Zihuan Zhang, Peng Wang, and Ning Li},
  journal={arXiv preprint arXiv:2502.19892},
  year={2025}
}
```


