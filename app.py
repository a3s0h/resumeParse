import PyPDF2
import re
from flask import Flask, render_template, request
from cleanResume import cleanResume
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the category mapping dictionary
with open('category_mapping.pkl', 'rb') as file:
    category_mapping = pickle.load(file)

@app.route('/')
def show_upload_page():
    return render_template('upload.html')

@app.route('/', methods=['GET', 'POST'])
def process_upload():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']

        if pdf_file.filename == '':
            return render_template('upload.html', upload_error='Please upload a file.')

        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        # Clean the extracted text using cleanResume function
        cleaned_text = cleanResume(text)
        print(cleaned_text)
        # Load the TfidfVectorizer used for feature extraction
        with open('vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)

        # Preprocess the input text and transform it into features
        cleaned_text = cleanResume(text)
        features = vectorizer.transform([cleaned_text])

        # Load the model
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)

        # Make predictions using the loaded model
        probabilities = model.predict_proba(features)

        # Get the indices of the top three probabilities in descending order
        top_indices = np.argsort(probabilities[0])[::-1][:3]

        # Get the corresponding category labels
        top_categories = [category_mapping[i] for i in top_indices]

        # Combine category labels and probabilities into a list of tuples
        top_predictions = list(zip(top_categories))

        return render_template('upload.html', predicted_categories=top_categories)


    return render_template('upload.html')


if __name__ == "__main__":
    app.run(debug=True)
