<div align="center">

<h1>BiMoGen</h1>
<h3>Bidirectional Motion-Text Generation<br>via Unified Masked Discrete Diffusion</h3>

<p>
  <a href="https://wengwanjiang.github.io/">Wanjiang Weng</a><sup>1,*</sup>&emsp;
  <a href="https://yongliang-wu.github.io/">Yongliang Wu</a><sup>1,*</sup>&emsp;
  <a href="https://xiaofeng-tan.github.io/">Xiaofeng Tan</a><sup>1</sup>
  <br>
  <a href="https://scholar.google.com/citations?user=PpqUnAwAAAAJ&amp;hl=en">Xingyu Zhu</a><sup>2</sup>&emsp;
  <a href="https://scholar.google.com/citations?user=z_4-QfQAAAAJ&amp;hl=en">Wenbo Zhu</a><sup>3</sup>&emsp;
  <a href="https://hongsong-wang.github.io/">Hongsong Wang</a><sup>1,&dagger;</sup>
</p>

<p>
  <sup>1</sup>Southeast University&emsp;
  <sup>2</sup>National University of Singapore&emsp;
  <sup>3</sup>HeyGen
  <br>
  <sup>*</sup>Equal contribution&emsp;
  <sup>&dagger;</sup>Corresponding author
</p>

<!-- Fill in HUGGINGFACE_URL. -->
<p>
  <a href="https://neurips.cc/virtual/2026/loc/atlanta/poster/152019"><img src="https://img.shields.io/badge/NeurIPS-2026-6842C2?style=flat-square" alt="NeurIPS 2026"></a>
  <a href="https://wengwanjiang.github.io/BiMoGen-Page/"><img src="https://img.shields.io/badge/Project-Web-4A90D9?style=flat-square&amp;logo=googlechrome&amp;logoColor=white" alt="Project Web"></a>
  <a href="HUGGINGFACE_URL"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&amp;logo=huggingface&amp;logoColor=black" alt="Hugging Face"></a>
  <a href="https://github.com/wengwanjiang/BiMoGen"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&amp;logo=github&amp;logoColor=white" alt="GitHub"></a>
  <a href="https://arxiv.org/abs/2609.35407"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=flat-square&amp;logo=arxiv&amp;logoColor=white" alt="arXiv"></a>
</p>

<p>
  <a href="#-quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#-dependencies">Dependencies</a> &nbsp;·&nbsp;
  <a href="#-data-preparation">Data Preparation</a> &nbsp;·&nbsp;
  <a href="#-training">Training</a> &nbsp;·&nbsp;
  <a href="#-evaluation">Evaluation</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

</div>

**BiMoGen** unifies text-to-motion generation and motion-to-text captioning through masked discrete diffusion.

## 🚩 News

- **2026/10/20** — Code and pretrained models released.
- **2026/09/26** — BiMoGen is accepted to **[NeurIPS 2026](https://neurips.cc/virtual/2026/loc/atlanta/poster/152019)**! 🎉

## 📋 TODO List

- [x] Environment setup
- [x] Data preparation and motion token encoding
- [x] Training code
- [x] Inference code
- [x] Evaluation code
- [x] Pretrained models and assets

## ⚡ Quick Start

<details>
<summary>Installation, downloads, and inference</summary>

### 1. Installation

```bash
git clone https://github.com/wengwanjiang/BiMoGen.git
cd BiMoGen

conda create -n bimogen python=3.10 -y
conda activate bimogen
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download models and assets

Download the [pretrained model](https://drive.google.com/drive/folders/1GyKW8VQGrVQdPfnqrqDdruHVLk4dWpqC?usp=sharing) and [assets](https://drive.google.com/drive/folders/1GyKW8VQGrVQdPfnqrqDdruHVLk4dWpqC?usp=sharing), then extract them in the repository root:

```bash
tar -xzf bimogen-85000.tar.gz
tar -xzf bimogen-assets.tar.gz
```

The model is saved under `checkpoints/bimogen-85000/` and the assets under `deps/`.

### 3. Text to motion

```bash
bash scripts/inference.sh \
  --text "A person walks forward and turns left." \
  --frames 120 --output output/walk.npy
```

The generated motion is saved to `output/walk.npy` as HumanML3D features with shape `(120, 263)` at 20 FPS.

### 4. Motion to text

```bash
bash scripts/inference.sh --motion output/walk.npy
```

The generated caption is printed to the terminal.

</details>

## 📦 Dependencies

<details>
<summary>Pretrained assets and MotionGPT VQ-VAE</summary>

### deps

```text
deps/
├── motiongpt/
│   └── motiongpt_s3_h3d.tar     # MotionGPT VQ-VAE for motion encoding and decoding
├── t2m/t2m/
│   ├── VQVAEV3_CB1024_CMT_H1024_NRES3/meta/
│   │   ├── mean.npy            # VQ-VAE normalization mean
│   │   └── std.npy             # VQ-VAE normalization standard deviation
│   ├── Comp_v6_KLD005/
│   │   ├── opt.txt             # Text-motion evaluator configuration
│   │   └── meta/
│   │       ├── mean.npy        # Evaluator normalization mean
│   │       └── std.npy         # Evaluator normalization standard deviation
│   └── text_mot_match/model/
│       └── finest.tar          # Pretrained text-motion matching evaluator
└── glove/
    ├── our_vab_data.npy        # GloVe word vectors for evaluation
    ├── our_vab_words.pkl       # Evaluator vocabulary
    └── our_vab_idx.pkl         # Word-to-vector index mapping
```

### MotionGPT VQ-VAE

We use the VQ-VAE from [MotionGPT](https://github.com/OpenMotionLab/MotionGPT). To download it separately from the assets bundle:

```bash
mkdir -p deps/motiongpt
curl -L --fail --retry 3 \
  https://huggingface.co/OpenMotionLab/MotionGPT-base/resolve/main/motiongpt_s3_h3d.tar \
  -o deps/motiongpt/motiongpt_s3_h3d.tar
```

The checkpoint is saved to `deps/motiongpt/motiongpt_s3_h3d.tar`. Evaluator assets are also available through [MotionGPT's preparation instructions](https://github.com/OpenMotionLab/MotionGPT#-quick-start), and GloVe through [MoMask](https://github.com/EricGuo5513/momask-codes/blob/main/prepare/download_glove.sh).

</details>

## 🗂️ Data Preparation

<details>
<summary>HumanML3D and motion token encoding</summary>

### 1. HumanML3D

Follow [HumanML3D](https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data) to prepare the dataset, then place it under `datasets/HumanML3D/`:

```bash
mkdir -p datasets
ln -s /absolute/path/to/HumanML3D datasets/HumanML3D
```

```text
datasets/HumanML3D/
├── train.txt
├── val.txt
├── test.txt
├── new_joint_vecs/             # HumanML3D motion features (T, 263)
└── texts/                      # Text annotations
```

### 2. Encode motion tokens

We normalize HumanML3D features with the MotionGPT VQ-VAE statistics and encode them into motion tokens:

```bash
bash scripts/encode_motion.sh --splits train test --device cuda
```

The tokens are saved to:

```text
datasets/HumanML3D/motiongpt_vq_codes_train.pt
datasets/HumanML3D/motiongpt_vq_codes_test.pt
```

</details>

## 🏋️ Training

<details>
<summary>Pretraining and instruction tuning</summary>

Set data paths and training options in [configs/bimogen.yaml](configs/bimogen.yaml), then run both stages on eight GPUs:

```bash
bash scripts/train.sh
```

For a single GPU:

```bash
accelerate launch --num_processes 1 -m training.train \
  --config configs/bimogen.yaml --stage pretrain

accelerate launch --num_processes 1 -m training.train \
  --config configs/bimogen.yaml --stage sft \
  --resume output/bimogen/pretrain-final
```

</details>

## 📊 Evaluation

<details>
<summary>Text-to-motion and motion-to-text evaluation</summary>

Evaluate both tasks:

```bash
bash scripts/evaluate.sh
```

Or run each task separately:

```bash
# Text-to-motion
python -m evaluation.eval_t2m \
  --checkpoint checkpoints/bimogen-85000 \
  --seed 42 --output-json output/evaluation/t2m.json

# Motion-to-text
python -m evaluation.eval_m2t \
  --checkpoint checkpoints/bimogen-85000 \
  --seed 42 --output-json output/evaluation/m2t.json
```

</details>

## Acknowledgments

We thank [LLaDA](https://github.com/ML-GSAI/LLaDA), [MotionGPT](https://github.com/OpenMotionLab/MotionGPT), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [pycocoevalcap](https://github.com/salaniz/pycocoevalcap), and [BERTScore](https://github.com/Tiiiger/bert_score) for their open-source contributions.

## Citation

```bibtex
@inproceedings{weng2026bimogen,
  title     = {{BiMoGen}: Bidirectional Motion-Text Generation via Unified Masked Discrete Diffusion},
  author    = {Weng, Wanjiang and Wu, Yongliang and Tan, Xiaofeng and Zhu, Xingyu and Zhu, Wenbo and Wang, Hongsong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
