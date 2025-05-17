import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Dataset
data = pd.read_csv("combine.csv", low_memory=False, encoding='utf-8', on_bad_lines="skip")
data.columns = data.columns.str.strip()

# Ensure Label column exists
if 'Label' not in data.columns:
    raise ValueError("The 'Label' column is missing from the dataset.")

# Define attack labels as malicious
malicious_labels = ['DDoS', 'PortScan']  # Add all attack-specific labels here
data['Category'] = data['Label'].apply(lambda x: 'Malicious' if x in malicious_labels else 'Benign')

# Prepare features and labels
feature_columns = [col for col in data.columns if col != 'Label' and col != 'Category']
numeric_features = data[feature_columns].select_dtypes(include=[np.number]).columns
features = data[numeric_features].copy()

# Replace infinite values and handle outliers
features = features.replace([np.inf, -np.inf], np.nan)  # Replace inf values with NaN
features = features.fillna(features.mean())  # Replace NaN values with column-wise mean

# Optional: Cap extreme values (outliers) to a reasonable range
for col in features.columns:
    lower_limit = features[col].quantile(0.01)  # 1st percentile
    upper_limit = features[col].quantile(0.99)  # 99th percentile
    features[col] = np.clip(features[col], lower_limit, upper_limit)

# Encode labels
labels = data['Category']
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Load trained model
model = load_model(r"C:\Users\Nilay\Downloads\cnn model\cnn model\model.h5")

# Predict using the trained model
predictions = model.predict(features_scaled)
predicted_classes = np.argmax(predictions, axis=1)

# Map predictions back to class names
# Predict using the trained model
predictions = model.predict(features_scaled)
predicted_classes = np.argmax(predictions, axis=1)

# Map predictions back to class names
known_labels = label_encoder.classes_  # Retrieve known labels from LabelEncoder
class_names = []

for pred_class in predicted_classes:
    if pred_class < len(known_labels):
        class_names.append(known_labels[pred_class])
    else:
        class_names.append("Unknown")  # Handle unseen labels gracefully

# Filter for malicious predictions
malicious_indices = [i for i, label in enumerate(class_names) if label == 'Malicious']
malicious_packets = data.iloc[malicious_indices]

# Generate iptables rules for malicious packets
print("\nApplying iptables rules for malicious packets...")
for _, flow in malicious_packets.iterrows():
    try:
        # Blocking based on Destination Port
        dst_port = int(flow['Destination Port'])
        rule = f"iptables -A INPUT -p tcp --dport {dst_port} -j DROP"
        os.system(rule)
        print(f"Rule applied: {rule}")
    except KeyError:
        print("Error: Missing 'Destination Port' in dataset. Skipping flow...")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Summary of malicious traffic blocked
attack_counts = malicious_packets['Label'].value_counts()
print("\nSummary of Blocked Malicious Traffic:")
print(attack_counts)

# Monitor iptables rules
print("\nCurrent iptables Rules:")
os.system("iptables -L -v -n")
