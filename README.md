## Aortic Aneurysm Analysis

This repository contains the implementation code for our published research on automated aortic aneurysm detection and volume estimation using deep learning approaches.

**📄 Published Paper**: [*Automated Aortic Aneurysm Detection and Volume Estimation Using Deep Learning*](https://www.mdpi.com/2075-4418/15/14/1804) - *Diagnostics* 2024, 15(14), 1804.

This repository adapts Meta's **Segment Anything Model 2 (SAM2)** for medical imaging applications, specifically for detecting and segmenting aortic aneurysms in CT scans. Our implementation combines U-Net-based aorta localization, SAM2 video segmentation, and machine learning approaches for automated aneurysm localization and volume measurements.

## 🔬 About This Project

This work builds upon Meta's SAM2 foundation model to create a specialized pipeline for aortic aneurysm analysis:

1. **U-Net Aorta Localization**: Uses a trained U-Net model to identify aorta coordinates in CT slices
2. **SAM2 Video Tracking**: Applies SAM2's video segmentation capabilities to track aortic regions across sequential CT slices
3. **Automated Boundary Detection**: Implements both expert systems and LSTM-based approaches to identify aneurysm boundaries
4. **Comprehensive Evaluation**: Provides comparison tools between automated methods and human annotations

## 📋 Key Features

- **Medical Image Processing**: Specialized DICOM handling and 8-bit conversion for CT scans
- **Multi-Modal Approach**: Combines traditional computer vision (U-Net) with foundation models (SAM2)
- **Aneurysm Detection**: Automated identification of aneurysm start/end boundaries
- **Performance Evaluation**: Cross-validation and comparison metrics against human annotations
- **Visualization Tools**: Comprehensive plotting and analysis capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.5.1+
- CUDA-compatible GPU (recommended)
- Required Python packages (see [Installation](#installation))

### Installation

1. **Clone this repository:**
   ```bash
   git clone <your-repository-url>
   cd Sam2-Repository
   ```

2. **Install SAM2 dependencies:**
   ```bash
   cd sam2
   pip install -e .
   pip install -e ".[notebooks]"
   ```

3. **Install additional requirements:**
   ```bash
   pip install tensorflow scikit-image pydicom natsort matplotlib xlsxwriter scikit-learn
   ```

4. **Download SAM2 checkpoints:**
   ```bash
   cd sam2/checkpoints
   ./download_ckpts.sh
   cd ../..
   ```

### Usage

**All code is available in**: `aneurysm analysis/notebooks/aneurysm.ipynb`

This notebook contains the complete implementation including:
- U-Net for aorta localization
- SAM2 video tracking  
- LSTM and Expert System for boundary detection
- Performance evaluation and visualization

**Pretrained U-Net Model**: For the trained U-Net model file, contact: `alirezab@email.sc.edu`

## 🔧 Pipeline Overview

1. **U-Net**: Locates aorta coordinates in CT slices
2. **SAM2**: Tracks aortic regions across sequential slices  
3. **Boundary Detection**: Expert system and LSTM approaches
4. **Evaluation**: Cross-validation and correlation analysis

## 🏥 Applications

### Example Data Structure

data structure follows like:
```
aorta/
├── aneurysm_jpg/              # JPG converted CT slices
│   ├── patient-001/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   ├── labels.txt             # Ground truth annotations
│   └── aorta_loc_full.txt     # Aorta coordinates from U-Net
├── sam2_pred/                 # SAM2 prediction outputs
└── aorta_unet_model.h5       # Trained U-Net model
```

```
M2S/              
├── patient-001/
│   ├── 0.dcm
│   ├── 1.dcm
│   └── ...
├── patient-002/
│   ├── 0.dcm
│   ├── 1.dcm
│   └── ...
```
## 🔧 Model Components

### 1. U-Net Aorta Localization
- Identifies aorta coordinates in CT slices
- Filters slices with sufficient segmented pixels (threshold: 600)
- Outputs center coordinates for SAM2 initialization

### 2. SAM2 Video Segmentation
- Uses SAM2's video prediction capabilities
- Tracks aortic regions across sequential CT slices
- Implements retry mechanism for robust segmentation

### 3. Boundary Detection Methods

#### Expert System Approach
- Rule-based detection using pixel count analysis
- Window-based comparison for abnormality detection
- Identifies regions with 20% increase/decrease in segmentation

#### LSTM Approach
- Deep learning model with bidirectional LSTM layers
- Input: Sequential segmented pixel counts (200 frames)
- Output: Binary mask indicating aneurysm boundaries
- Architecture: Input LSTM → FC layers → Output LSTM → Sigmoid

## 📊 Evaluation Metrics

The repository includes comprehensive evaluation tools:

- **Cross-validation**: 5-fold cross-validation for LSTM model
- **Correlation Analysis**: R² correlation with human annotations
- **Boundary Accuracy**: Start/end position comparison
- **Volume Analysis**: Segmented pixel volume correlation


This pipeline is designed for:

- **Aortic Aneurysm Screening**: Automated detection in CT scans
- **Longitudinal Monitoring**: Tracking aneurysm progression over time  
- **Clinical Decision Support**: Providing quantitative measurements
- **Research Applications**: Large-scale aneurysm analysis studies

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{aneurysm2024diagnostics,
  title={Automated Aortic Aneurysm Detection and Volume Estimation Using Deep Learning},
  journal={Diagnostics},
  volume={15},
  number={14},
  pages={1804},
  year={2024},
  publisher={MDPI},
  doi={10.3390/diagnostics15141804},
  url={https://www.mdpi.com/2075-4418/15/14/1804}
}
```

## 🙏 Acknowledgments

### SAM2 Attribution

This project is built upon **Segment Anything Model 2 (SAM2)** developed by Meta's FAIR team:

- **Original Repository**: [facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
- **Paper**: [SAM 2: Segment Anything in Images and Videos](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)
- **Authors**: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, and many others at Meta AI

**Citation for SAM2:**
```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```

### Third-Party Components

- **Connected Components**: GPU-based algorithm adapted from [`cc_torch`](https://github.com/zsef123/Connected_components_PyTorch) (BSD 3-Clause License)

## 📄 License

This project respects the licensing terms of its dependencies:

- **SAM2 Components**: Licensed under [Apache 2.0](https://github.com/facebookresearch/segment-anything-2/blob/main/LICENSE)
- **Connected Components (`cc_torch`)**: BSD 3-Clause License (see `LICENSE_cctorch`)
- **This Project's Modifications**: Please specify your preferred license for the medical imaging adaptations

## ⚠️ Disclaimer

This software is for research purposes only and has not been approved for clinical use.
