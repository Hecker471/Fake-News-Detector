import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Convert text data into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["statement"])  # Convert statements to numerical data
y = df["label"]  # Labels (1 = Real, 0 = Fake)

# Show the shape of the transformed data
print(f"Feature matrix shape: {X.shape}")  # (num_samples, num_features)
print(f"First feature vector: {X[0].toarray()}")  # Example of first transformed statement
