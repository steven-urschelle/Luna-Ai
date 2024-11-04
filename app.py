from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from save_model import save_to_csv  # Import your save function

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
    global texts, labels, model, X  # Declare globals at the top of the function

    user_input = request.form['user_input']
    user_label = request.form['user_label']  # Optional label input

    user_input_vectorized = vectorizer.transform([user_input])
    
    # Make a prediction
    prediction = model.predict(user_input_vectorized)[0]

    # Convert prediction to string for a valid response
    prediction_str = str(prediction)

    # Save new data if a label is provided
    if user_label:
        save_to_csv(user_input, user_label)  # Use your save function

        # Re-train the model with the updated data
        data = pd.read_csv(data_file)
        texts = data['text']
        labels = data['label']
        X = vectorizer.fit_transform(texts)
        model = MultinomialNB()
        model.fit(X, labels)

    return prediction_str  # Return the prediction as a string

if __name__ == "__main__":
    app.run(debug=True)
