Week 3: Variational Autoencoders (VAE)
Date: 28-01-2026

Overview
This week focuses on implementing and training a Variational Autoencoder (VAE), a powerful generative model that learns to encode data into a latent space and decode it back to generate new samples.

Notebook: Week_03_28_01_2026.ipynb
Key Tasks
Task 1: Dataset Preparation
Loaded the Fashion-MNIST dataset with appropriate transformations
Configured data loaders for training and testing with batch size 64
Normalized pixel values to [0, 1] range using ToTensor()
Task 2: Reparameterization Trick
Implemented the reparameterization trick to sample from the latent distribution
Formula: z = mu + std * epsilon where epsilon is sampled from standard normal distribution
Enables gradient flow through the stochastic sampling layer
Task 3: Loss Function
Implemented the ELBO (Evidence Lower Bound) loss combining two components:
Reconstruction Loss: Binary Cross Entropy between original and reconstructed images
KL Divergence Loss: Regularizes the latent space distribution toward N(0,1)
Loss formula: BCE + KLD = BCE - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
Task 4: Model Training
Trained the VAE for 20 epochs using Adam optimizer (learning rate: 1e-3)
Used batch size of 64 and latent dimension of 2 (for easy visualization)
Tracked and plotted training loss per epoch
Task 5: Sample Generation
Generated new synthetic fashion items from random noise vectors
Compared original test images with VAE reconstructions
Visualized quality of generation and reconstruction
Task 6: Latent Space Visualization
Plotted the 2D latent space with color-coded fashion item classes
Demonstrated how the VAE organizes different clothing types in the latent space
Shows smooth transitions between different fashion categories
Model Architecture
Encoder
Input: 28×28 flattened image (784 features)
Hidden layer: 400 neurons with ReLU activation
Output: Mean (mu) and Log-Variance (log_var) vectors of latent dimension 2
Decoder
Input: Latent vector z (2 dimensions)
Hidden layer: 400 neurons with ReLU activation
Output: 28×28 image (784 features) with Sigmoid activation to constrain to [0, 1]
Hyperparameters
Batch Size: 64
Learning Rate: 1e-3
Epochs: 20
Latent Dimension: 2
Optimizer: Adam
Results & Observations
The VAE successfully learns to reconstruct fashion items
Generated samples show realistic clothing patterns
Latent space visualization reveals organized representation of fashion classes
KL divergence regularization prevents posterior collapse and maintains meaningful latent structure
Technologies & Libraries
PyTorch: Deep learning framework
TorchVision: Fashion-MNIST dataset and transforms
Matplotlib & NumPy: Visualization and numerical operations
CUDA: GPU acceleration (if available)
