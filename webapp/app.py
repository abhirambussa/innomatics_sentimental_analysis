from flask import Flask, render_template, request
import pickle
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK data if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

app = Flask(__name__, static_url_path='/static')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define the clean function for preprocessing
def clean(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

# Instantiate a CountVectorizer with the clean function as preprocessor
vect = CountVectorizer(preprocessor=clean)

def preprocess_input(text):
    cleaned_text = clean(text)
    return cleaned_text

# Load the fitted vectorizer
with open('model/vectorizer.pkl', 'rb') as file:
    vect = pickle.load(file)

# Load the trained model
model = joblib.load("model/naive_bayes_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    a = request.form.get("input")
    preprocessed_input = preprocess_input(a)
    new_input_dtm = vect.transform([preprocessed_input])

    prediction = model.predict(new_input_dtm)

    # Convert prediction to human-readable form
    prediction_label = "positive" if prediction[0] == 1 else "negative"

    return render_template("output.html", prediction=prediction_label)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
