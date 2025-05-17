# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split

# # Step 1: Load the dataset and clean the column names
# df = pd.read_csv("combine.csv", low_memory=False)

# # Strip any leading/trailing spaces from column names
# df.columns = df.columns.str.strip()

# # Step 2: Check for infinite or non-numeric values
# print("Data types:\n", df.dtypes)

# # Identify non-numeric columns
# non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
# print("Non-numeric columns:", non_numeric_columns)

# # Step 3: Convert non-numeric columns to numeric (coerce errors to NaN)
# df_clean = df.apply(pd.to_numeric, errors='coerce')

# # Check for any NaN values after conversion
# print("Missing values after conversion:\n", df_clean.isna().sum())

# # Step 4: Handle infinite values and very large values
# print("Checking for infinite values...")
# print(np.isinf(df_clean).sum())

# # Replace infinite values with NaN
# df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

# # Check for missing values after replacing infinite values
# print("Missing values after replacing infinite values:\n", df_clean.isna().sum())

# # Step 5: Handle NaN values (fill with mean or median)
# df_clean.fillna(df_clean.mean(), inplace=True)

# # Optionally, apply a threshold to handle large outliers
# threshold = 1e6  # Set this according to your dataset characteristics
# df_clean[df_clean > threshold] = threshold

# # Check if there are still NaN values after thresholding
# print("Missing values after thresholding:\n", df_clean.isna().sum())

# # Fill any remaining NaN values
# df_clean.fillna(df_clean.mean(), inplace=True)

# # Step 6: Separate features and labels
# X = df_clean.drop("Label", axis=1)  # Features
# y = df_clean["Label"]  # Labels

# # Step 7: Encode labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Step 8: Normalize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Step 9: Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# # Print a preview of the cleaned and normalized data
# print("First 5 rows of X_train:\n", X_train[:5])
# print("First 5 labels of y_train:\n", y_train[:5])
