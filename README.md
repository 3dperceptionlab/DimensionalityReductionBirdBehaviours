<h1 align="center">Optimizing IoT Video Data: Dimensionality Reduction for Efficient Deep Learning on Edge Computing</h1>

This repository contains the official implementation for: **"Optimizing IoT Video Data: Dimensionality Reduction for Efficient Deep Learning on Edge Computing"**. It implements state-of-the-art dimensionality reduction techniques tailored for analyzing bird behaviors from video data, enabling efficient processing on resource-constrained environments like edge computing devices.

## Key Highlights

- **Dimensionality Reduction Methods**:
  - **Feature Embeddings**: Extract embeddings from pre-trained video models like Swin Transformer, reducing data size by over 6,000 times while maintaining high classification accuracy.
  - **Autoencoders**: Compress video data using spatio-temporal autoencoders, achieving significant reductions in data size.
  - **Single-Frame Analysis**: Process individual frames using CNNs, Vision Transformers, or DINO+HoG, leveraging spatial features for classification.

- **Dataset**:
  - **Visual WetlandBirds Dataset**: Includes videos of bird behaviors such as feeding, preening, and swimming, annotated with species and actions. The dataset is described in the [dataset paper](https://arxiv.org/abs/2501.08931) and available on [GitHub](https://github.com/3dperceptionlab/Visual-WetlandBirds).

## Repository Structure

```plaintext
├── Single_Frame_CNN_Transformer/    # Implementation of single-frame analysis using CNNs and Vision Transformers
├── Single_Frame_Dino_HoG/           # Implementation of single-frame analysis using DINO and HoG features
├── features/                        # Feature embedding extraction from pre-trained video models
├── reduction_autoencoder/           # Implementation of the autoencoder method
├── LICENSE                          
├── README.md                        
