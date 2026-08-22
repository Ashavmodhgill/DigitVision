# DigitVision

DigitVision is an interactive handwritten-digit classifier built with PyTorch and Streamlit. Draw a digit on the canvas or upload an image, and the app preprocesses it into the MNIST format before predicting a class from 0 to 9.

The project combines a compact convolutional neural network, a reproducible training notebook, and a focused browser interface for real-time inference.

## Features

- Draw digits directly in the Streamlit canvas
- Upload PNG, JPG, or JPEG digit images
- Automatic grayscale conversion, inversion, resizing, and normalization
- CNN inference on CPU or CUDA when available
- Predicted digit with confidence score
- Ranked confidence board for all ten classes
- Processed 28 x 28 input preview
- Model status and inference-device information in the sidebar
- Local model and dataset files protected by `.gitignore`

## Project Structure

```text
DigitVision/
|-- app.py                    # Streamlit inference application
|-- DigitVision.ipynb         # Dataset preparation, training, evaluation, export
|-- requirements.txt          # Python dependencies
|-- .gitignore                # Local files excluded from Git
|-- digitvision_model.pth    # Trained checkpoint, kept local and ignored
`-- Digit/                    # MNIST files, kept local and ignored
```

## Model Architecture

The classifier accepts a single-channel 28 x 28 MNIST image and uses three convolutional blocks:

| Stage | Operation | Output channels |
| --- | --- | ---: |
| 1 | Conv2d, ReLU, MaxPool2d | 32 |
| 2 | Conv2d, ReLU, MaxPool2d | 64 |
| 3 | Conv2d, ReLU, MaxPool2d | 128 |
| Classifier | Flatten, Linear, ReLU, Linear | 10 classes |

The final dense layers map the extracted features from `3 x 3 x 128` to 256 hidden units and then to the ten digit classes.

## Training Details

The training workflow is in `DigitVision.ipynb`:

- Dataset: MNIST
- Training examples: 60,000
- Input transform: tensor conversion and normalization with mean `0.5` and standard deviation `0.5`
- Batch size: 64
- Optimizer: Adam
- Learning rate: `0.001`
- Loss function: CrossEntropyLoss
- Epochs: 10
- Recorded test accuracy: `98.72%`

The final notebook cell saves a checkpoint containing the model state dictionary and evaluation accuracy:

```python
torch.save(checkpoint, "digitvision_model.pth")
```

## Requirements

- Python 3.10 or newer
- A virtual environment is recommended
- CPU works by default; CUDA is used automatically when available

## Installation

In PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation for the current session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## Prepare the Model

The trained checkpoint is intentionally ignored by Git. After cloning the repository, open `DigitVision.ipynb` in VS Code or Jupyter and run the training and export cells. This creates:

```text
digitvision_model.pth
```

The Streamlit app expects this file in the same directory as `app.py`.

## Run the App

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Then open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

For the best result, draw one centered digit with a clear stroke, or upload a close-cropped digit image.

## Inference Pipeline

1. Capture a drawing or uploaded image.
2. Convert the image to grayscale.
3. Resize it to 28 x 28 pixels.
4. Invert the pixels when the background is brighter than the digit.
5. Scale pixels to the range 0 to 1.
6. Apply MNIST-style normalization.
7. Run the image through the CNN.
8. Convert logits to probabilities with softmax.
9. Display the highest-probability digit and all class confidences.

## Git Safety

The repository ignores local and generated files that should not be committed:

- Trained model files such as `.pth`, `.pt`, and `.ckpt`
- MNIST data under `Digit/`
- Virtual environments
- Environment variables and Streamlit secrets
- Notebook checkpoints
- Python caches and generated outputs

Source code, requirements, the training notebook, and this README remain available for version control.

## Troubleshooting

### Model status says "Weights not found"

Run the final export cell in `DigitVision.ipynb` and confirm that `digitvision_model.pth` is beside `app.py`.

### Drawing canvas is unavailable

Install the dependencies again and restart Streamlit:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

### Predictions are weak

Use a centered, high-contrast digit with minimal background noise. The model was trained on MNIST-style 28 x 28 handwritten digits, so heavily cropped, rotated, or decorated images may reduce accuracy.

## Technologies

- Python
- PyTorch
- Torchvision
- Streamlit
- Streamlit Drawable Canvas
- NumPy
- Pillow

## License

No license has been specified for this project yet.
