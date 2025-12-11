# Learn from Foundation Model: Fruit Detection Model without Manual Annotation
## Segmentation-Description-Matching-Distilling




[![Paper](https://img.shields.io/static/v1?label=Paper&message=Github&color=red)](https://doi.org/10.1016/j.patcog.2025.112799) [![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue)](https://github.com/AgRoboticsResearch/SDM-D.git) [![Dataset](https://img.shields.io/static/v1?label=Dataset&message=Kaggle&color=green)](https://www.kaggle.com/datasets/mmwang0/megafruits)
[![Colab](https://img.shields.io/static/v1?label=Demo&message=Colab&color=orange)](https://colab.research.google.com/drive/1tmgaogh9VjGp4sJnBMcUSyKrMfcuqwDw?usp=drive_link) 

![SDM-D architecture](./asset/1-all2.png)

**🍄Segmentation-Description-Matching-Distilling** is a framework designed to distill small models that enable panoramic perception of complex agricultural scenes from foundation models without relying on manual labels. At its core is SDM, which operates without pre-training or significant resource consumption, within a segment-then-prompt paradigm. SDM demonstrates strong zero-shot performance across various fruit detection tasks (object detection, semantic segmentation, and instance segmentation), consistently outperforming SOTA OVD methods across various fruit perception tasks, demonstrating superior dexterity and generality.

## 🔥Colab try
We provide a Google's Colab example [![Colab](https://img.shields.io/static/v1?label=Demo&message=Colab&color=orange)](https://colab.research.google.com/drive/1tmgaogh9VjGp4sJnBMcUSyKrMfcuqwDw?usp=drive_link), where anyone can use our project quickly and easily.

## 🍇Installation
### 1. Prepare the environment
First, install PyTorch suitable for your machine, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
conda create -n SDM python=3.10
conda activate SDM
pip install torch torchvision # install the torch you need
git clone  https://github.com/AgRoboticsResearch/SDM-D.git
cd SDM-D
pip install -r requirements.txt
```

### 2. Install Segment-Anything-2 model
Please install the Segment-Anything-2 model first.

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

### 3. Install OpenCLIP
Please install OpenCLIP.
```bash
pip install open_clip_torch
```

## 🚀Getting Started

### Download Checkpoints

1) First, we need to download the model weight file to the `./checkpoint` folder. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints
./download_ckpts.sh
```

The model in SDM is: [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt), you can also download this only.

2) The OpenCLIP can be utilized with `open_clip.create_model_and_transforms`,  and the model name and corresponding pretrained keys are compatible with the outputs of open_clip.list_pretrained().

```bash
import open_clip
open_clip.list_pretrained()
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
```

### Label prediction


(1) Our project is very easy to use, just need to run SDM.py. 

First, please put `your dataset` into `./Images` folder, there is an example (image.jpg is also okay):
```bash
Images/
├── your_dataset_name/
│   ├── train/
│   │   ├── 001.png
│   │   ├── 002.png
│   │   └── ...
│   ├── val/
│   │   ├── 012.png
│   │   ├── 050.png
│   │   └── ...
```

Second, put your desrctiptions and labels into a .txt file, you can put it in `./description` folder. Each line is in the format of `description text, label`.

Third, please give parameters, you can run:

```bash
cd SDM-D

python SDM.py --image_folder /path/to/images --out_folder /path/to/output --des_file /path/to/prompt.txt

```
In the last, the structure of the `output` folder is as follows:
```bash
output/
│── mask/  # mask of the instance segmentation task
│── labels/  # label of the instance segmentation task in YOLO format
│── mask_idx_visual/ # visual the mask ids 
│── mask_color_visual/  # visual masks with color [need to set, see follows (2)]
│── label_box_visual/  # visual detection boxed of masks [need to set, see follows (3)]
│── json/  # json of the instance segmentation task [need to set, see follows (4)]
```
(2) If you want to get colorful visual results, you need to set the `mask_color_visual` as `Ture`. The visual results will be saved in `out_folder/mask_color_visual` folder.

```bash
python SDM.py --image_folder /path/to/images --out_folder /path/to/output --des_file /path/to/prompt.txt --mask_color_visual True
```
![SDM-D architecture](./asset/mask_visual.png)

(3) If you want to visual the detection boxes, you need to set `'--box_visual'` as `True` or run:
```bash
python SDM.py --image_folder /path/to/images --out_folder /path/to/output --des_file /path/to/prompt.txt --box_visual True
```
(4) If you want to see the detail of masks, you can save their `.josn` file by set the `'--save_json'` as `True`:

```bash
python SDM.py --image_folder /path/to/images --out_folder /path/to/output --des_file /path/to/prompt.txt --save_json True
```

(5) If you want to explore parameters that fit your own dataset, you can try `../notebook/SDM.ipynb`.


### Label conversion

(1) If you want to get object detection lables, just run:

```bash
python ../seg2label/seg2det.py
```

(2) If you want to get semantic segmentation labels, just run:

```bash
python ../seg2label/seg2semantic.py
```

(3) If you want to get labels for specific kinds of objects, you can abstract their labels just run:

```bash
python ../seg2label/abstract.py
```
    
### The design of description texts

The design of prompts greatly affects the model performance, particularly in tasks involving fine-grained distinctions. We summarize an effective prompt template: `a/an {color} {shape} {object} with {feature}`, where the color description is the most crucial. Here is some examples of the prompt design:

![Prompt design](./asset/prom-end34.png)

Although some error can be avoided by adding a new description (e.g., Fig.(c)"black background"), considering the generality of the entire dataset, We don't recommend it. Regarding the design of the number of prompt texts, we recommend that users consider the characteristics of objects within the entire scene. While an excessive number of prompts may lead to higher accuracy, it can adversely affect the model's generalization ability, rendering it less suitable for large-scale datasets and requiring a lot of time and effort.

## 🌻SDM-D

### Distillation

These pseudo-labels generated by SDM can serve as supervision for small, edge-deployable models (students), bypassing the need for costly manual annotation. The SDM-D is highly versatile and model-agnostic, with no restrictions on the choice of the student model. Any compact model optimized for a downstream task can be seamlessly integrated into the distillation process. And the is no distillation loss in SDM-D, all the distilled models have better accuracy. And the distilled can achive better performence with few-shot learning.
![few-shot](./asset/few-shot.png)

### Model Description


#### Comparison of Inference Time and GPU Memory Allocation for Each Method

|     | **Grounded SAM** | **YOLO-World** | **SDM** | **SDM-D (YOLOv8s)** |
|-------------------------|------------------|----------------|---------|---------------------|
|Inference Time (ms)| 8,090.81         | 99.32         | 7,615.08         | **18.96**         |
| **GPU Memory Allocation (MiB)** | 7,602         | 2,268         | 6,650         | **878**         |
|                                                                  |

## 📖Dataset

We introduce a high-quality, comprehensive fruit instance segmentation dataset named [[`MegaFruits`](https://www.kaggle.com/datasets/mmwang0/megafruits)]. This dataset encompasses 20,242 images of strawberries with 569,382 pseudo masks, 2,400 manually labeled images of yellow peaches with 10,169 masks, and 2,540 manually labeled images of blueberries with 20,656 masks. Leveraging the capabilities of our method, we are able to generate such a large scale of pseudo-segmentation labels. We anticipate this resource will catalyze further research and practical advancements in agricultural vision systems.

## 💘Acknowledgements

- [SAM 2](https://github.com/facebookresearch/sam2.git)
- [OpenCLIP](https://github.com/mlfoundations/open_clip.git)
  
  
## 🔥Contributors

Our project wouldn't be possible without the contributions of these amazing people! Thank you all for making this project better.

![Contribution](./asset/people3.png)

