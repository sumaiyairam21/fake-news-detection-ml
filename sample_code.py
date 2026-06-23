from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

text = ["fake news", "real news"]
labels = [0, 1]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text)

model = LogisticRegression()
model.fit(X, labels)

print("Model trained")
