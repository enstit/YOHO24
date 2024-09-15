<div align="center">
  <img src="./assets/sound-wave.png" width="225px">
  <h1 align="center">🔉 YOHO24</h2>
</div>

> [!NOTE]
> This repository is part of the final exam for the *Deep Learning* course held by professor Alessio Ansuini and Emanuele Ballarin ([`@emaballarin`](https://github.com/emaballarin)) at [University of Trieste](https://www.units.it/en) in the 2023-2024 academic year.

In this project we - Davide Capone ([`@davidecapone`](https://github.com/davidecapone)) and Enrico Stefanel ([`@enstit`](https://github.com/enstit)), Master's student at [University of Trieste](https://www.units.it/en) - try to improve the performance of **YOHO** algorithm for Audio Segmentation and Sound Event Detection by applying features introduced in more recent YOLO models.

## Abstract
Following the introduction of You-Only-Look-Once (YOLO) model in 2015[^1], which quickly revolutionized the way the Object Detection problem was approached due to its extremely lightweight structure but that offered very good accuracy, many researchers focused their work on such models family. In the following years new versions of YOLO were presented[^2][^3][^5][^7][^8][^9], which further lightened the model and made it more reliable. Some other studies, instead, concentrated on applying YOLO concepts to data types other than visual ones: this is the case of the YOHO algorithm (You-Only-Hear-Once)[^6], which targets the Audio Segmentation and Sound Event Detection problems. In this project we propose to select the improvements applied to YOLO in the latest versions, and apply them to the YOHO structure to see if these bring significant improvements to the model, testing it on the same datasets as the YOHO paper and comparing the two.

## Usage

### Datasets
The datasets used in this project are (almost) the same ones used in [the original YOHO paper](https://doi.org/10.48550/arXiv.2109.00962) to evaluate its performances, with the exception of the *18h of audio from BBC Radio Devon* dataset (namely, the TUTSoundEvent Detection dataset for the third task of the DCASE challenge 2017 and the Urban Sound Event Detection dataset).

The [`yoho/dataset.py`](./yoho/dataset.py) script automatically downloads raw data and process it. Simply run
```python
python3 -m yoho.dataset [--urbansed] [--tut]
```
specifying wheter if you want to download `urbansed` dataset or `tut` dataset.

Regarding the relevant licenses, please refer to the individual sources.

### Train
Model weight are available in the [`models`](./models) folder. If you want to train your personal version of `YOHO`, modify the [`yoho/train.py`](./yoho/train.py) script and start the training with
```python
python3 -m yoho.train [--name YourCustomModelName] [--epochs 50] [--batch-size 32] [--cosine-annealing]
```
It is also possible to pass the `[--spec-augment]` parameter if you want to augment your training dataset with SpecAugment augmentations[^4].

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.


[^1]: Redmon, J., Divvala, S., Girshick, R., and Farhadi, A., "You Only Look Once: Unified, Real-Time Object Detection", *arXiv e-prints*, 2015. [doi:10.48550/arXiv.1506.02640](https://doi.org/10.48550/arXiv.1506.02640).

[^2]:
    Redmon, J. and Farhadi, A., "YOLO9000: Better, Faster, Stronger", *arXiv e-prints*, 2016. [doi:10.48550/arXiv.1612.08242](https://doi.org/10.48550/arXiv.1612.08242).

[^3]:
    Redmon, J. and Farhadi, A., "YOLOv3: An Incremental Improvement", *arXiv e-prints*, 2018. [doi:10.48550/arXiv.1804.02767](https://doi.org/10.48550/arXiv.1804.02767).

[^4]:
    Daniel S.-P., William C., Yu Z., Chung-Cheng C., Barret Z., Ekin D.-C., Quoc V.-L., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition", *arXiv e-prints*, 2019. [doi:10.48550/arXiv:1904.08779](https://doi.org/10.48550/arXiv.1904.08779).

[^5]:
    Bochkovskiy, A., Wang, C.-Y., and Liao, H.-Y. M., "YOLOv4: Optimal Speed and Accuracy of Object Detection", *arXiv e-prints*, 2020. [doi:10.48550/arXiv.2004.10934](https://doi.org/10.48550/arXiv.2004.10934).

[^6]:
    Venkatesh, S., Moffat, D., and Reck Miranda, E., "You Only Hear Once: A YOLO-like Algorithm for Audio Segmentation and Sound Event Detection", *arXiv e-prints*, 2021. [doi:10.48550/arXiv.2109.00962](https://doi.org/10.48550/arXiv.2109.00962).

[^7]:
    Ultralytics, "YOLOv5: A state-of-the-art real-time object detection system", 2021. [https://docs.ultralytics.com](https://docs.ultralytics.com).

[^8]:
    Li, C., "YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications", *arXiv e-prints*, 2022. [doi:10.48550/arXiv.2209.02976](https://doi.org/10.48550/arXiv.2209.02976).

[^9]:
    Wang, C.-Y., Bochkovskiy, A., and Liao, H.-Y. M., "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors", *arXiv e-prints*, 2022. [doi:10.48550/arXiv.2207.02696](https://doi.org/10.48550/arXiv.2207.02696).
