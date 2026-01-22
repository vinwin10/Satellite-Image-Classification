# Satellite Image Classification: From CNNs to Hybrid ViTs

## Project Overview
This Capstone Project focuses on the application of Deep Learning to classify satellite imagery for agricultural use. Acting as data scientists for a fertilizer company seeking to expand into new territories, the objective was to automate the identification of agricultural versus non-agricultural land.

Moving beyond simple classification, this project explores the evolution of computer vision architectures, comparing traditional **Convolutional Neural Networks (CNNs)** in different frameworks against cutting-edge **Vision Transformers (ViTs)** and **CNN-ViT Hybrid** models.

##  Key Objectives
* **Binary Classification:** Accurately classify images into `class_0_non_agri` and `class_1_agri`.
* **Framework Comparison:** Evaluate the trade-offs between **Keras (TensorFlow)** for rapid prototyping and **PyTorch** for granular control.
* **Advanced Architectures:** Implement and train Vision Transformers (ViTs) and Hybrid models to leverage both local feature extraction and global attention mechanisms.
* **Efficiency:** Implement lazy loading and data augmentation to handle large datasets on limited memory resources.

##  Tech Stack
* **Language:** Python 3.x
* **Notebooks:** Jupyter
* **Deep Learning Frameworks:**
  * **PyTorch** (`torch`, `torch.nn`, `torch.utils.data`)
  * **Keras / TensorFlow** (`tensorflow.keras`, `layers.MultiHeadAttention`)
* **Data Manipulation:** NumPy (for bulk loading and array manipulation)
* **Data Processing:** Custom data loaders, Lazy loading techniques

##  Project Structure
The project is divided into four distinct modules, mirroring the learning path from data ingestion to advanced model deployment:

### Module 1: Data Preparation & Exploration 
* **Data Handling:** Implemented "lazy loading" to handle satellite imagery efficiently, reading images one by one to avoid memory overflows.
* **Preprocessing:** Resized images (e.g., $64 \times 64$ pixels) and converted them to NumPy arrays.
* **Augmentation:** Applied techniques to balance the dataset (equal Agri and Non-Agri samples) and increase dataset diversity.

### Module 2: Building Basic Classifiers (CNNs) 
* **Keras Implementation:** Built a beginner-friendly CNN using high-level Keras commands. Used standard layers (`Conv2D`, `MaxPooling`, `Dense`) to extract edges and colors.
* **PyTorch Implementation:** Built a custom class inheriting from `nn.Module`. This provided a "hands-on" approach to defining the forward pass and training loops, offering deeper debugging capabilities.
* **Metrics:** Evaluated using Binary Cross-Entropy Loss and Accuracy.

### Module 3: Advanced Models with Vision Transformers (ViTs) 
* **Concept:** Treated images as sequences of patches (similar to words in NLP models like GPT), using **Self-Attention** to focus on specific regions (e.g., green vegetation patches).
* **Hybrid Architecture:** Designed a **CNN-ViT Hybrid**.
    * *CNN Component:* Extracts local features efficiently.
    * *ViT Component:* Analyzes global correlations and complex patterns.
* **Transfer Learning:** Utilized pre-trained CNN backbones to save computational resources.

### Module 4: Integration & Evaluation 
* **Performance:** The Hybrid models achieved **>99% accuracy**.
* **Validation:** Utilized cross-validation and confusion matrices to minimize false positives/negatives.
* **Regularization:** Implemented techniques like **Dropout** to prevent overfitting (memorization of training data).

##  Key Results
* **Accuracy:** The final CNN-ViT Hybrid model achieved over **95% accuracy** on the test set.
* **Comparison:** While Keras offered faster initial setup, PyTorch provided better flexibility for customizing the Hybrid architecture.
* **ViT Performance:** Vision Transformers excelled at distinguishing subtle vegetation patterns that standard CNNs occasionally missed, though they required careful hyperparameter tuning.

##  Lessons Learned
* **Lazy Loading is crucial:** Loading all satellite images into RAM simultaneously is a recipe for system crashes; batch processing is essential.
* **No "Best" Framework:** Keras is unbeatable for quick proofs-of-concept, while PyTorch is superior for research and complex custom architectures.
* **The Power of Hybrids:** Combining the inductive bias of CNNs with the dynamic attention of Transformers results in robust models suitable for complex "real-world" textures like satellite imagery.
