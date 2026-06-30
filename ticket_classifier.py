import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv("support_tickets.csv")

print("Dataset Preview:")
print(data.head())

# -------------------------------
# CATEGORY CLASSIFICATION
# -------------------------------
X = data["Ticket"]
y = data["Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

category_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

category_model.fit(X_train, y_train)

category_predictions = category_model.predict(X_test)

print("\nCategory Classification Accuracy:")
print(accuracy_score(y_test, category_predictions))

print("\nCategory Classification Report:")
print(classification_report(y_test, category_predictions))

# -------------------------------
# PRIORITY PREDICTION
# -------------------------------
y_priority = data["Priority"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    y_priority,
    test_size=0.2,
    random_state=42
)

priority_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

priority_model.fit(X_train2, y_train2)

priority_predictions = priority_model.predict(X_test2)

print("\nPriority Prediction Accuracy:")
print(accuracy_score(y_test2, priority_predictions))

print("\nPriority Classification Report:")
print(classification_report(y_test2, priority_predictions))

# -------------------------------
# TEST A NEW TICKET
# -------------------------------
new_ticket = [
    "My payment has failed and money was deducted."
]

predicted_category = category_model.predict(new_ticket)
predicted_priority = priority_model.predict(new_ticket)

print("\nNew Ticket:")
print(new_ticket[0])

print("Predicted Category:", predicted_category[0])
print("Predicted Priority:", predicted_priority[0])