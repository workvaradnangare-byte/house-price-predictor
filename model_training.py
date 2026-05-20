import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from pickle import dump

# Load dataset
data = pd.read_csv("case_Study_6.csv")

# One Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded = encoder.fit_transform(data[["furnishing_status"]])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["furnishing_status"])
)

# Merge encoded data
data = data.drop("furnishing_status", axis=1)
data = pd.concat([data, encoded_df], axis=1)

# Features and target
features = data.drop("rent", axis=1)
target = data["rent"]

# Train model
model = LinearRegression()
model.fit(features, target)

print("Model trained successfully!")

# User Inputs
bhk = float(input("Enter BHK: "))
size = float(input("Enter size (sqft): "))
bathroom = float(input("Enter number of bathrooms: "))
furnishing = input(
    "Enter furnishing (Furnished / Semi-Furnished / Unfurnished): "
)

# Encode input
encoded_input = encoder.transform(
    pd.DataFrame([[furnishing]], columns=["furnishing_status"])
)

encoded_input_df = pd.DataFrame(
    encoded_input,
    columns=encoder.get_feature_names_out(["furnishing_status"])
)

# Create input dataframe
input_data = pd.DataFrame([{
    "bhk": bhk,
    "size_sqft": size,
    "bathroom": bathroom
}])

# Merge encoded columns
input_data = pd.concat([input_data, encoded_input_df], axis=1)

# Ensure all columns match training data
input_data = input_data.reindex(columns=features.columns, fill_value=0)

# Prediction
prediction = model.predict(input_data)

print("Predicted Rent:", int(prediction[0]))

# Save model
with open("Model.pkl", "wb") as f:
    dump(model, f)

# Save encoder
with open("Encoder.pkl", "wb") as f:
    dump(encoder, f)

print("Model and Encoder saved successfully!")