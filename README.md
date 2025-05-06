# Semi-Supervised Learning through GANs for Melanoma Detection

This project implements a semi-supervised learning approach using Generative Adversarial Networks (GANs) for melanoma detection in skin lesion images.

## Project Structure
```
.
├── data/                    # Data directory
│   ├── labeled/            # Labeled melanoma images
│   └── unlabeled/          # Unlabeled images
├── models/                 # Model implementations
│   ├── generator.py       # Generator network
│   ├── discriminator.py   # Discriminator network
│   └── gan.py            # GAN implementation
├── utils/                 # Utility functions
│   ├── data_loader.py    # Data loading and preprocessing
│   └── visualization.py   # Visualization utilities
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data:
   - Place labeled melanoma images in `data/labeled/`
   - Place unlabeled images in `data/unlabeled/`

2. Train the model:
```bash
python train.py
```

3. Evaluate the model:
```bash
python evaluate.py
```

## Model Architecture

The project uses a semi-supervised GAN architecture with:
- A Generator network that generates synthetic skin lesion images
- A Discriminator network that performs both real/fake classification and melanoma detection

## Results

The model achieves improved melanoma detection accuracy by leveraging both labeled and unlabeled data through the semi-supervised learning approach.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
