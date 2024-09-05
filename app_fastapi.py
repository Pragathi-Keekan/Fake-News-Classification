from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download NLTK resources
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load data
news_df = pd.read_csv('News.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming and tokenization function
ps = PorterStemmer()


def tokenize_and_stem(content):
    tokens = word_tokenize(content)
    tokens = [ps.stem(word) for word in tokens if word.isalpha() and word.lower() not in stopwords.words('english')]
    return ' '.join(tokens)


# Apply stemming and tokenization function to content column
news_df['content'] = news_df['content'].apply(tokenize_and_stem)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)

# Fit Naive Bayes model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, Y_train)

# Fit SVM model
svm_model = LinearSVC()
svm_model.fit(X_train, Y_train)

# Fit Gradient Boosting model
gradient_boost_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gradient_boost_model.fit(X_train, Y_train)


# Function to calculate metrics
def calculate_metrics(model, X_train, Y_train, X_test, Y_test):
    train_accuracy = accuracy_score(Y_train, model.predict(X_train))
    test_accuracy = accuracy_score(Y_test, model.predict(X_test))
    train_precision = precision_score(Y_train, model.predict(X_train))
    train_recall = recall_score(Y_train, model.predict(X_train))
    train_f1 = f1_score(Y_train, model.predict(X_train))
    test_precision = precision_score(Y_test, model.predict(X_test))
    test_recall = recall_score(Y_test, model.predict(X_test))
    test_f1 = f1_score(Y_test, model.predict(X_test))

    return train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
def read_item(request: Request):
    return templates.TemplateResponse("index_fastapi.html", {"request": request})


@app.post("/predict")
def predict(request: Request, input_text: str = Form(...), selected_model: str = Form(...)):
    if input_text:
        # Choose the selected model for prediction
        if selected_model == 'Logistic Regression':
            model = logistic_model
        elif selected_model == 'Naive Bayes':
            model = naive_bayes_model
        elif selected_model == 'SVM':
            model = svm_model
        elif selected_model == 'Gradient Boosting':
            model = gradient_boost_model

        # Make prediction
        prediction = model.predict(vector.transform([tokenize_and_stem(input_text)]))

        # Display result
        result_text = f'{selected_model} Prediction: {"Fake" if prediction[0] == 1 else "Real"}'

        # Calculate and display metrics (if applicable)
        train_accuracy, train_precision, train_recall, train_f1, test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(
            model, X_train, Y_train, X_test, Y_test)

        # Prepare metrics data
        metrics_data = [
            ('Accuracy', format(train_accuracy, '.4f'), format(test_accuracy, '.4f')),
            ('Precision', format(train_precision, '.4f'), format(test_precision, '.4f')),
            ('Recall', format(train_recall, '.4f'), format(test_recall, '.4f')),
            ('F1 Score', format(train_f1, '.4f'), format(test_f1, '.4f')),
        ]

        return templates.TemplateResponse("result_fastapi.html",
                                          {"request": request, "result": result_text, "metrics": metrics_data})
    else:
        return templates.TemplateResponse("index_fastapi.html",
                                          {"request": request, "error": "Please enter a news article."})
