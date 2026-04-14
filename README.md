<div align="center">
<img src="./docs/_static/facil_logo.png" width="100px">

# Ouroboros - A continual learning paradigma.

---

<p align="center">
  <a href="#what-is-ouroboros">What is Ouroboros</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="src/approach#approaches-1">Approaches</a> •
  <a href="src/datasets#datasets">Datasets</a> •
  <a href="src/networks#networks">Networks</a> •
  <a href="#license">License</a> •
  <a href="#cite">Cite</a>
</p>
</div>

---

## What is Ouroboros
Ouroboros started as code for the paper:  
_**Continual learning with pretrained models**_  
*Chionas Ioannis*  
([Platform](https://link-to-paper))

It was build upon the existing Framework for Analysis of Class Increamental Learning (FACIL) and was heavily inspired by PYCIL.
It combines the logic of the two frameworks, ulilizing Task Increamental Learning support using FACIL's head removal approach and the more 
up to date PYCIL trainer + Config logic. Reproducing the paper's experiments is supported plus adding/ suggesting new methods is more than welcome.
Expanding and building upon existing foundation is crucial for the development of science so feel free to use any part of this framework you find usefull.
If you do so please take the time to cite the framework: citing notes

## Key Features
This framework by deafault supports both task and class increamental learning due to the list of heads FACIL logic.

| Setting                                                                             | task-ID at train time | task-ID at test time | # of tasks |
| ----------------------------------------------------------------------------------- | --------------------- | -------------------- | ---------- |
| [class-incremental learning](https://ieeexplore.ieee.org/abstract/document/9915459) | yes                   | no                   | ≥1         |
| [task-incremental learning](https://ieeexplore.ieee.org/abstract/document/9349197)  | yes                   | yes                  | ≥1         |
| non-incremental supervised learning                                                 | yes                   | yes                  | 1          |

Current available approaches include:
<div align="center">
<p align="center"><b>
  • Finetuning
  • Freezing
  • Joint
  • LwF
  • iCaRL
  • EWC
  • PathInt
  • MAS
  • RWalk
  • EEIL
  • LwM
  • DMC
  • BiC
  • LUCIR
  • IL2M
  • SimpleCIL
  • LWF-DT
  • LWF-DT-COS
</b></p>
</div>

## How To Use
Clone this github repository:
```
git clone https://github.com/yiannischionas/ouroboros.git
cd ouroboros
```

<details>
  <summary>Optionally, create an environment to run the code (click to expand).</summary>

  ### Using a requirements file
  The library requirements of the code are detailed in [requirements.txt](requirements.txt). You can install them
  using pip with:
  ```
  python3 -m pip install -r requirements.txt
  ```

  ### Using a conda environment
  Development environment based on Conda distribution. All dependencies are in `environment.yml` file.

  #### Create env
  To create a new environment check out the repository and type: 
  ```
  conda env create --file environment.yml --name FACIL
  ```
  *Notice:* set the appropriate version of your CUDA driver for `cudatoolkit` in `environment.yml`.

  #### Environment activation/deactivation
  ```
  conda activate FACIL
  conda deactivate
  ```

</details>

To run the basic code:
```
python3 -u src/main_incremental.py
```
More options are explained in the [`src`](./src), including GridSearch usage. Also, more specific options on approaches,
loggers, datasets and networks.

### Scripts
We provide scripts to reproduce the specific scenarios proposed in 
_**Class-incremental learning: survey and performance evaluation**_:

* CIFAR-100 (10 tasks) with ResNet-32 without exemplars
* CIFAR-100 (10 tasks) with ResNet-32 with fixed and growing memory
* _MORE COMING SOON..._

All scripts run 10 times to later calculate mean and standard deviation of the results.
Check out all available in the [scripts](scripts) folder.

## License
Please check the MIT license that is listed in this repository.

## Cite
If you want to cite the framework feel free to use this preprint citation while we await publication:
```bibtex
@article{masana2022class,
  title={Class-Incremental Learning: Survey and Performance Evaluation on Image Classification},
  author={Masana, Marc and Liu, Xialei and Twardowski, Bartłomiej and Menta, Mikel and Bagdanov, Andrew D. and van de Weijer, Joost},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  doi={10.1109/TPAMI.2022.3213473},
  year={2023},
  volume={45},
  number={5},
  pages={5513-5533}}
}
```

---

The basis of FACIL is made possible thanks to [Chionas Yiannis](https://github.com/yiannischionas) but most imporantly the fareworks that inspired it:
FACIL
PYCIL
Feel free to contribute or propose new features by opening an issue!