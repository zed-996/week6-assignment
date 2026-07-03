# MNIST Image Denoising using Autoencoders

## Problem Statement
Build a deep learning model that can remove noise from images using an autoencoder architecture, trained and evaluated on the MNIST handwritten digits dataset.

## Dataset
Uses the **mnist_png** dataset — MNIST digits stored as 28×28 grayscale PNG files organized by label:
```
mnist_png/
├── training/   (60,000 images across folders 0-9)
└── testing/    (10,000 images across folders 0-9)
```

## Approach
We train autoencoders to **denoise** images:
1. **Add Gaussian noise** to clean MNIST images (noise factor = 0.5)
2. Feed the noisy images to the autoencoder **encoder**, which compresses them into a latent representation
3. The **decoder** reconstructs clean images from this representation
4. The model is trained using **MSE loss** between the denoised output and the original clean image

This forces the autoencoder to learn robust features that capture the essential structure of digits while ignoring noise.

## Three Model Architectures

### 1. Feed-Forward Neural Network (FFNN) Autoencoder
- Flattens 28×28 images into 784-dim vectors
- **Encoder**: 784 → 256 → 64 → 32 (ReLU activations)
- **Decoder**: 32 → 64 → 256 → 784 (ReLU + final Sigmoid)
- Custom weight initialization: Normal(0, 1/√n)

### 2. Transpose CNN Autoencoder
- Works directly on 2D image data (preserves spatial structure)
- **Encoder**: Conv2d(1→16) → MaxPool → Conv2d(16→4) → MaxPool → 4×7×7
- **Decoder**: ConvTranspose2d(4→16) → ConvTranspose2d(16→1) → 1×28×28
- Note: Transpose convolutions can produce checkerboard artifacts

### 3. Upsampled CNN Autoencoder
- Same encoder as Model 2
- **Decoder**: Uses nearest-neighbor upsampling + standard Conv2d + BatchNorm
- Avoids the checkerboard artifacts of transpose convolutions
- Generally produces smoother reconstructions

## Requirements
```
pip install -r requirements.txt
```

## How to Run
Open the notebook and run all cells:
```bash
jupyter notebook denoising_autoencoder.ipynb
```

This will:
1. Load MNIST images from the `mnist_png` folder
2. Train all 3 models for 20 epochs each
3. Save trained models to `models/`
4. Display and save visualization results to `results/`

## Project Structure
```
myProject/
├── denoising_autoencoder.ipynb  # Main notebook
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── models/                      # Saved model weights
├── results/                     # Output visualizations
│   ├── loss_curves.png
│   ├── FFNN_Autoencoder_denoising.png
│   ├── TransposeCNN_Autoencoder_denoising.png
│   └── UpsampledCNN_Autoencoder_denoising.png
```

## Key Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Epochs | 20 |
| Learning Rate | 0.001 |
| Noise Factor | 0.5 |
| Optimizer | Adam |
| Loss Function | MSE |
| Encoding Dim (FFNN) | 32 |

## Why MSE Loss?
Since we're comparing pixel values between the denoised output and original clean images, MSE (Mean Squared Error) is the natural choice — it's a regression task measuring per-pixel reconstruction quality.
