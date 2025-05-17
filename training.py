import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
data = pd.read_csv("combine.csv", low_memory=False, encoding='utf-8', on_bad_lines="skip")
data.columns = data.columns.str.strip()

if 'Label' not in data.columns:
    raise ValueError("The 'Label' column is missing from the dataset.")

# Prepare features and labels
feature_columns = [col for col in data.columns if col != 'Label']
numeric_features = data[feature_columns].select_dtypes(include=[np.number]).columns

# Basic cleaning
features = data[numeric_features].copy()
features = features.replace([np.inf, -np.inf], np.nan)
features = features.fillna(features.mean())

# Prepare labels
labels = data['Label']
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, labels_encoded, 
    test_size=0.2, 
    random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights for imbalanced classes
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = {i: total_samples / (len(class_counts) * count) 
                for i, count in enumerate(class_counts)}

# Define model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Add early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train model
print("\nTraining model...")
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert encoded labels back to original names
y_test_original = label_encoder.inverse_transform(y_test)
y_pred_original = label_encoder.inverse_transform(y_pred_classes)

# Calculate overall accuracy
accuracy = accuracy_score(y_test_original, y_pred_original)
print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_original, y_pred_original))

# Print sample predictions
print("\nSample Predictions:")
num_samples = min(20, len(y_test_original))  # Show up to 20 samples
indices = np.random.choice(len(y_test_original), num_samples, replace=False)

for idx in indices:
    print(f"Actual: {y_test_original[idx]}, Predicted: {y_pred_original[idx]}")

# Show confusion matrix
plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test_original, y_pred_original)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Print most common misclassifications
print("\nMost Common Misclassifications:")
misclassified_pairs = [(true, pred) for true, pred in zip(y_test_original, y_pred_original) if true != pred]
if misclassified_pairs:
    misclass_df = pd.DataFrame(misclassified_pairs, columns=['True', 'Predicted'])
    misclass_counts = misclass_df.groupby(['True', 'Predicted']).size().reset_index(name='count')
    misclass_counts = misclass_counts.sort_values('count', ascending=False)
    print(misclass_counts.head(10))
else:
    print("No misclassifications found!")

model.save(r"C:\Users\Nilay\Downloads\cnn model\cnn model\model.h5")

