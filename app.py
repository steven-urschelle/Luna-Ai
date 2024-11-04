from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import os

# Initialize Flask app
app = Flask(__name__)

# Load your model and vectorizer
data_file = 'data/data.csv'
data = pd.read_csv(data_file)
texts = data['text']
labels = data['label']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form['user_input']
    user_label = request.form['user_label']  # Assume the label is sent from the frontend
    user_input_vectorized = vectorizer.transform([user_input])
    
    # Make a prediction
    prediction = model.predict(user_input_vectorized)[0]

    # Update data.csv if the user provides a label
    if user_label:
        new_data = pd.DataFrame([[user_input, user_label]], columns=['text', 'label'])
        new_data.to_csv(data_file, mode='a', header=False, index=False)

        # Re-train the model with the updated data
        global texts, labels, model, X
        data = pd.read_csv(data_file)
        texts = data['text']
        labels = data['label']
        X = vectorizer.fit_transform(texts)
        model = MultinomialNB()
        model.fit(X, labels)

    return prediction

if __name__ == "__main__":
    app.run(debug=True)
