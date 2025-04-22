# Jeongak Jeongganbo Dataset

This is the official code repository for following paper
> On the Automatic Recognition of Jeongganbo Music Notation: Dataset and Approach
> [https://dl.acm.org/doi/10.1145/3715159](https://dl.acm.org/doi/10.1145/3715159)

In which we:
* Propose the first-ever OMR framework for Jeongganbo (정간보) notation.
* Propose the novel encoding system for Jeongganbo notation system.
* Propose a synthetic dataset of (jeonggan image, annotation) pairs and the generation method for end-to-end OMR.
* Achieve **89% accuracy** through synthetic data and augmentation.
* Create the **new dataset** dedicated to Korean court music in Jeongganbo notation.


## Overview

Jeonggan notation is a traditional Korean music notation system used in Korean classical music (gugak). This project focuses on synthesizing Jeonggan images with random contents and building an end-to-end OMR system to transcribe Jeonggan notation into the proposed encoding format.

The repository includes:
1. Tools for downloading and preprocessing Jeongganbo scorebooks
1. Synthetic Jeonggan image generator
1. PyTorch-based OMR model for recognizing Jeonggan content
1. Evaluation and inference tools

### Jeonggan Synthesizer
* Generates images of single Jeonggans containing random content
* Supports various configuration options for controlling the generation process:
  - Character variants
  - Noise application
  - Random symbol insertion
  - Layout element manipulation

### Works in progress
* [ ] Jeonggan drawer notebook
* [ ] KR<->EN Translation


## Installation

```bash
# Clone the repository
git clone git@github.com:MALerLab/jeongganbo-omr.git
cd jeongganbo-omr

# Install dependencies
pipenv --python 3.8.10
pipenv sync
```


## Project Structure

```
jeongganbo-omr/
├── configs/                            # Configuration files
├── dataset/                            # Dataset storage (Check the Releases page!)
|── checkpoints/                        # For model checkpoints (Check the Releases page!)
├── jngbomr/                            # Main package
│   ├── __init__.py
│   ├── data_utils.py                   # Data utilities
│   ├── inferencer.py                   # Inference utilities
│   ├── jeonggan_utils/                 # Jeonggan-related utilities
│   │   ├── __init__.py
│   │   ├── const.py                    # Constants definitions
│   │   ├── jeonggan_processor.py       # Mostly defrecated
│   │   ├── jeonggan_synthesizer.py
│   │   ├── jeongganbo_drawer.py
│   │   └── jeongganbo_reader.py
│   ├── model_zoo.py                    # Model definitions
│   ├── preprocess_utils/               # Preprocessing utilities
│   ├── trainer.py                      # Training utilities
│   ├── train_utils.py                  # Training helper functions
│   └── vocab_utils.py                  # Vocabulary utilities
├── outputs/                            # Training outputs
├── evaluate.py                         # Evaluation script
├── preprocess_jeongganbo_scorebooks.py # download needed data
├── run_omr_jeonggans.py                # Run OMR on individual jeonggans
├── run_omr_scores.py                   # Run OMR on full scores
└── train.py                            # Training script
```

## Usage

### Preprocessing Jeongganbo Scorebooks
```bash
# Before running this,
# please download dataset and checkpoints from the releases page!
python preprocess_jeongganbo_scorebooks.py
```

This script:
1. Downloads scorebooks from the Korean National Gugak Center
2. Splits PDFs into page images
3. Repairs page images (fixes formatting issues)
4. Splits pages into individual Jeonggan units
5. Prepares symbols for the synthesizer

### Generating Synthetic Jeonggan Images
```python
from jngbomr import JeongganSynthesizer, get_img_paths

# Get paths to note/symbol images
img_paths = get_img_paths('dataset/jeongganbo/synth/', ['notes', 'symbols'])

# Initialize synthesizer
synth = JeongganSynthesizer(img_paths)

# Generate a random Jeonggan
label, img = synth(
  char_variant=True,    # Use character variants
  apply_noise=True,     # Apply noise to the image
  random_symbols=True,  # Add random symbols
  layout_elements=True  # Apply layout elements
)

# Generate with specific label
label = "황:1 중:3 청황:5"
img = synth.generate_image_by_label(
  label, 
  width=100, 
  height=140, 
  char_variant=True, 
  apply_noise=True
)
```

### Training an OMR Model
```bash
python train.py
```

You can customize the training process by modifying the configuration in `configs/config.yaml`.

### Running OMR on Jeonggan Images
```bash
python run_omr_jeonggans.py
```

### Running OMR on Full Scores
```bash
python run_omr_scores.py
```

## License
This project is licensed under the MIT License.