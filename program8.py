# Program 8 – Sentiment Analysis


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    'review': [
        'This movie is excellent',
        'I really enjoyed this film',
        'Amazing movie and great acting',
        'This movie is fantastic',
        'I loved this movie',
        'This movie is terrible',
        'I hated this film',
        'Worst movie ever',
        'Very boring and disappointing',
        'The movie was bad'
    ],
    'sentiment': [
        'positive', 'positive', 'positive', 'positive', 'positive',
        'negative', 'negative', 'negative', 'negative', 'negative'
    ]
}

df = pd.DataFrame(data)

X = df['review']
y = df['sentiment']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

new_reviews = [
    "This movie is wonderful",
    "The film was very disappointing"
]

new_reviews_vectorized = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_vectorized)

print("\nSentiment Predictions:")
for review, sentiment in zip(new_reviews, predictions):
    print(review, ":", sentiment)
