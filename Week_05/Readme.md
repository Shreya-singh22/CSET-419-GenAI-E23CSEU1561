# 🖼️ Week 5 – Image Transformation Models  
**Date:** 11-02-2026  

This week focuses on practical implementation of deep learning models for image transformation tasks.  
We explored both:

- Building a **custom CNN Encoder–Decoder** for image colorization  
- Using a **pre-trained diffusion model (InstructPix2Pix)** for text-guided image editing  

Notebook: `Week_05_11_02_2026.ipynb`

---

# 🔹 Part I – Image Colorization using CNN Encoder–Decoder

## 📌 Objective
Train a neural network that takes a grayscale image as input and predicts its colored version.

---

## 🧩 Task 1 – Data Preparation

- Dataset used: **CIFAR-10**
- Images normalized to range **[-1, 1]**
- Batch size:
  - Training: 64
  - Testing: 10
- RGB images converted to grayscale for input
- Applied:
  - `transforms.ToTensor()`
  - `transforms.Normalize()`

The model learns to map:

```
Grayscale Image (1 channel) ➝ RGB Image (3 channels)
```

---

## 🏗️ Task 2 – Model Architecture

### 🔹 Encoder
Compresses input image into compact latent features.

- Conv2d layers:
  - 1 → 32 channels
  - 32 → 64 channels
- Stride = 2 (downsampling)
- Output size: **8 × 8 feature maps**

### 🔹 Decoder
Reconstructs the color image from latent features.

- ConvTranspose2d layers:
  - 64 → 32 channels
  - 32 → 3 channels
- Stride = 2 (upsampling)
- Final activation: `Tanh`
- Output range: **[-1, 1]**

---

## 🎯 Task 3 – Training

- Loss Function: **Mean Squared Error (MSE)**
- Optimizer: **Adam**
- Learning Rate: 0.001
- Epochs: 10

The model directly minimizes pixel-level reconstruction error between predicted and true RGB images.

---

## 📊 Task 4 – Evaluation

- Generated colorized outputs from grayscale inputs
- Compared:
  - Input (Grayscale)
  - Model Output (Predicted Color)
  - Ground Truth (Original Color)
- Visualized results side-by-side using Matplotlib

---

# 🔹 Part II – Text-Guided Image Editing with InstructPix2Pix

## 📌 Objective
Use a diffusion-based model to modify images based on natural language prompts.

---

## ⚙️ Task 1 – Model Setup

- Loaded **Stable Diffusion InstructPix2Pix** from Hugging Face
- Used:
  - `EulerAncestralDiscreteScheduler`
- Device selection:
  - CUDA (if available)
  - CPU (fallback)
- Safety checker disabled for research experimentation

---

## 🎛️ Task 2 – Parameter Tuning

Key generation controls:

- **IMAGE_STABILITY (default: 2.0)**  
  Controls how much the original image is preserved  

- **TEXT_STRENGTH (default: 10.0)**  
  Controls how strongly the prompt influences the output  

- **Inference Steps: 40**  
  Balance between quality and speed  

---

## 🧠 Task 3 – Prompt-Based Editing

Different types of edits were explored:

### 🌍 Environment Changes
- "Make me sit in open mountains"
- "Turn the background into a cyberpunk city"

### 🎨 Style Transformations
- "Make it look like a pencil sketch"

### 🎭 Creative Edits
- "Add two balloons on the head"
- "Make me drive a supercar"

### 🌙 Scene Modifications
- "Make me study under the moon"
- "Make me hospitalized in a hospital bed"

### 🏍️ Travel Scenarios
- "Make me ride a superbike"

---

## 🖼️ Task 4 – Output Generation

- Generated 512×512 images
- Saved outputs with descriptive filenames
- Displayed original vs edited image comparison

---

# 🧠 Key Concepts Learned

## Image Colorization
- Encoder–Decoder networks
- Latent feature representation
- Pixel-wise loss optimization (MSE)
- CNN spatial feature learning

## Diffusion-Based Editing
- Iterative denoising process
- Prompt-guided image modification
- Guidance scale tuning
- Leveraging large pre-trained models

---

# 🎯 Learning Outcomes

By the end of this week, I was able to:

- Build and train a CNN encoder–decoder model from scratch
- Perform grayscale-to-color image reconstruction
- Use Hugging Face Diffusers for image editing
- Control diffusion model behavior using guidance parameters
- Work efficiently with PyTorch and modern deep learning pipelines

---

# 🛠️ Technical Stack

- **PyTorch** – Custom model development  
- **Torchvision** – Dataset handling & preprocessing  
- **Diffusers (Hugging Face)** – Pre-trained diffusion models  
- **PIL (Pillow)** – Image handling  
- **Matplotlib** – Visualization  

---

⭐ This week provided hands-on experience with both traditional CNN architectures and modern generative diffusion models for real-world image transformation tasks.
