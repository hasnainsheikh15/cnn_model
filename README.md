# 🛡️ Malicious Packet Detection Using CNN

This project is a machine learning-based system designed to **detect malicious packets** flowing through public Wi-Fi networks. It uses a **Convolutional Neural Network (CNN)** to classify network traffic based on packet features, aiming to provide automated threat identification and enhanced public network security.

---

## Features

- Detects malicious vs. benign network packets
- Uses a custom-trained CNN model on extracted packet features
- Displays classification accuracy, confusion matrix, and common misclassifications
- Includes preprocessing steps for data cleaning and scaling
- Tracks model performance using early stopping and class balancing

---

## Model Architecture

The CNN model is a basic fully-connected neural network trained on numerical features from packet data. Key elements include:

- Input layer scaled with `StandardScaler`
- Dense hidden layers with ReLU activation and dropout
- Output layer with softmax for multi-class classification
- Loss function: `sparse_categorical_crossentropy`
- Optimizer: `Adam`

---

## Dataset

The model is trained using a CSV dataset (`combine.csv`) that contains labeled packet data. It assumes the following:

- **Numerical packet features** across various network metrics
- A **'Label'** column denoting the type of packet (e.g., benign, malicious)

---

## Output & Evaluation

- Accuracy Score
- Classification Report
- Confusion Matrix (visualized with Seaborn)
- Randomly sampled prediction outputs
- Most common misclassification pairs

---

## Installation

1. Install required libraries:

   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
   ```

2. Ensure the dataset (`combine.csv`) is placed in the root directory.

---

## How to Run

Run the training script:

```bash
python train_and_evaluate.py
```

This will:

- Train the CNN model
- Evaluate its performance
- Save the model to disk (`model.h5`)

---

## Project Structure

```
malicious-packet-detector-cnn/
│
├── combine.csv                 # Dataset (not included in repo)
├── train_and_evaluate.py      # Main training & evaluation script
├── model.h5                   # Saved CNN model (generated after training)
└── README.md
```

---

## 📌 Requirements

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- pandas, numpy, seaborn, matplotlib

---

## 🔒 License

This project is open-source under the **MIT License**.

---

## 🙋‍♂️ Author

**Hasnain**  
📧 _[mdhasnainsheikh15@gmail.com]_  
🔗 GitHub: [@hasnainsheikh15](https://github.com/hasnainsheikh15)
