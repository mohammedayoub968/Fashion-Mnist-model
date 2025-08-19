
# Fashion MNIST Classification with Neural Networks

## 📌 Project Overview
This project applies **Artificial Neural Networks (ANNs)** to classify images from the **Fashion MNIST dataset**.  
The goal is to predict clothing categories (e.g., shirts, shoes, bags) based on grayscale images.

---

## 🗂 Dataset
- **Fashion MNIST**: 70,000 images (60,000 for training, 10,000 for testing)
- Image size: **28x28 pixels**, grayscale
- 10 classes (e.g., T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

---

## ⚙️ Steps in the Project

### 1. **Data Preprocessing**
- Normalized pixel values (0–255 → 0–1 scaling)
- Split data into training, validation, and test sets

### 2. **Model Building**
- Used **Keras Tuner** with RandomSearch to optimize hyperparameters:
  - Number of hidden layers (`n_hidden_layers`)
  - Number of neurons per layer (`neurons`)
  - Activation function (`activation`)
  - Learning rate (`learning_rate`)

### 3. **Training**
- Best model trained for **15 epochs**
- Early stopping considered to reduce overfitting

### 4. **Evaluation**
- Training Accuracy: ~89%
- Validation Accuracy: ~88%
- Test Accuracy: **88.2%**

### 5. **Metrics**
- **Classification Report** (Precision, Recall, F1-score per class)
- **Confusion Matrix** to visualize misclassifications

---

## 📊 Results

- **Overall Accuracy**: ~88%
- Model performs best on classes like **Sneakers, Sandals, Ankle boots** (precision > 0.95)
- Struggles slightly with **Shirts & T-shirts** (lower precision/recall, ~0.75–0.83)

**Classification Report (excerpt):**
```
           precision    recall  f1-score
T-shirt       0.80      0.87      0.83
Trouser       0.97      0.98      0.98
Shirt         0.75      0.62      0.68
Sneaker       0.92      0.94      0.93
Ankle boot    0.94      0.96      0.95
```

**Confusion Matrix**: Shows that some shirts were misclassified as T-shirts or coats.

---

## 📚 Tools & Libraries
- **Python 3.11**
- **TensorFlow / Keras**
- **Keras Tuner**
- **Scikit-learn** (classification report, confusion matrix)
- **NumPy, Matplotlib, Seaborn** (visualization)

---

## 🚀 How to Run the Project
1. Clone this repository
2. Install required libraries:
   ```bash
   pip install tensorflow keras keras-tuner scikit-learn matplotlib seaborn
   ```
3. Run the training script
4. Evaluate performance on test set

---

## ✅ Conclusion
- ANN with hyperparameter tuning achieved **~88% accuracy** on Fashion MNIST.
- Main challenge: Differentiating similar clothing items (Shirts, T-shirts, Coats).
- Future improvements:
  - Use **Convolutional Neural Networks (CNNs)** for better image feature extraction
  - Apply **data augmentation** to improve generalization

---

📌 *Author: Mohammed Ayoub*  
📅 *2025*
