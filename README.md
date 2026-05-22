# Plant Disease Detection using VGG16

Developed by: **Moaz**

This project implements a robust deep learning solution to identify and classify plant leaf diseases from images. Using the VGG16 architecture and Transfer Learning, the model is trained to recognize multiple symptoms across various plant species, providing a valuable tool for early diagnosis and precision agriculture.

## 📌 Problem Statement
Plant diseases are a major threat to global food security. Manual identification is slow and requires significant botanical expertise. This project automates the detection process using Convolutional Neural Networks (CNN), allowing for early intervention and reduced crop loss for farmers.

## 📊 Dataset
The model utilizes the **New Plant Diseases Dataset** from Kaggle.
*   **Content:** Thousands of augmented images of healthy and diseased leaves.
*   **Scale:** Covers 38 different class labels (various fruit and vegetable diseases).
*   **Source:** (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## 🛠️ Technical Implementation
*   **Framework:** PyTorch
*   **Architecture:** VGG16 (Pre-trained on ImageNet)
*   **Preprocessing:** 
    *   Resizing images to 224 x 224 pixels.
    *   Random Horizontal Flips and Rotations (to improve generalization).
    *   Normalization of pixel values.
*   **Training Strategy:** 
    *   Frozen feature extraction layers to retain pre-trained knowledge.
    *   Custom Fully Connected (FC) layer replacement to match 38 plant disease classes.
    *   **Optimizer:** Adam (LR=0.0001).
    *   **Loss Function:** Cross-Entropy Loss.
*   **Data Split:** 80% Training, 20% Validation/Testing.

## 🚀 How to Run
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/plant-disease-detection.git
    cd plant-disease-detection
    ```
2.  **Dataset Setup:**
    The notebook uses `kagglehub` to download the dataset automatically via the notebook. Ensure you have your Kaggle API credentials configured if running locally.
3.  **Run the Notebook:**
    Launch `Plant_Disease_Detection_using_VGG16.ipynb` in an environment like Google Colab or Kaggle and run all cells.

## 📈 Evaluation Results
The model's performance is analyzed through:
*   **Accuracy & Loss Curves:** Visualization of training and validation progress.
*   **Class Distribution Analysis:** Using Matplotlib and Seaborn to visualize dataset balance.
*   **Inference Pipeline:** Includes a `predict_image` function to test the model on any new leaf scan.

## 📦 Requirements
*   `torch`
*   `torchvision`
*   `kagglehub`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `Pillow (PIL)`

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
