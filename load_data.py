import pandas as pd

# Load dataset
columns = ["id", "label", "statement", "subject", "speaker", "job", "state", "party", "barely_true", "false", "half_true", "mostly_true", "pants_on_fire", "context"]
df = pd.read_csv("liar_dataset/train.tsv", sep="\t", header=None, names=columns)

# Keep only the necessary colums
df = df[["statement", "label"]]

print(df["label"].unique())

# Convert labels to binary (1 = Real, 0 = Fake)
df["label"] = df["label"].map(lambda x: 1 if x in ["true", "mostly-true", "half-true"] else 0)

# Remove missing values
df = df.dropna()

# Show dataset size after cleaning
print(f"Dataset size after cleaning: {len(df)}")

# Show first few rows
print(df.head())

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
print("Cleaned dataset saved as cleaned_data.csv!")
