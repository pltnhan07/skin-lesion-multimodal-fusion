# skin-lesion-multimodal-fusion

# Multimodal Graph-Based Fusion for Skin Lesion Classification

PyTorch implementation of a multi-scale, hierarchical graph fusion module for skin lesion classification using clinical images and tabular metadata.

## Features
- Multi-scale visual feature extraction with ResNet-50
- Metadata encoding
- Hierarchical graph fusion with multi-head attention
- 5-fold cross-validation
- Evaluation with detailed metrics

## Usage

### 1. Install dependencies
pip install -r requirements.txt


### 2. Prepare datasets
- Place your training and test images and CSVs as described in the code.
- Update paths in `main.py` if needed.

### 3. Run training and evaluation
python main.py


## Citation

If you use this code, please cite:
Pham, N.L.T., et al., "A Multimodal Deep Ensemble Framework for Skin Lesion Classification." IUKM 2025.

## Contact

Nhan Le Thanh Pham  
Email: pltnhan07@gmail.com
