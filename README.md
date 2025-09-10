📧 Spam Mail Detection Project
📌 Overview

This project implements a Spam Mail Classifier using Machine Learning.
The classifier predicts whether an email message is:

📩 Ham (Not Spam)

🚫 Spam

This is a binary text classification problem widely used as a benchmark in Natural Language Processing (NLP).

📂 Dataset

The dataset consists of labeled emails/messages with two categories: ham and spam.

Preprocessing includes:

Removing stopwords & punctuation

Lowercasing text

Tokenization and lemmatization

Converting text to numerical features using Bag of Words (BoW) and TF-IDF

⚙️ Installation & Requirements

Install dependencies with:

pip install -r requirements.txt


requirements.txt

numpy
pandas
matplotlib
seaborn
scikit-learn
nltk

🚀 Project Workflow

Data Loading – Import dataset (CSV).

Exploratory Data Analysis (EDA) – Class distribution, word frequency, email length visualization.

Text Preprocessing – Clean and prepare text for modeling.

Feature Extraction – Convert emails into numerical features using CountVectorizer and TF-IDF.

Model Training – Train multiple ML models:

Naïve Bayes

Logistic Regression

Random Forest

Support Vector Machine (SVM)

Evaluation – Compare performance using:

Accuracy

Precision, Recall, F1-score

Confusion Matrix

📊 Results

Naïve Bayes with TF-IDF achieved the best performance with >95% accuracy.

Logistic Regression and SVM also performed well.

🖼️ Visualizations

Word clouds for spam vs ham

Distribution plots of message lengths

Confusion matrix for model performance

▶️ Usage

Run the Jupyter notebook:

jupyter notebook spammaildetection.ipynb


Or run as a Python script:

python spam_mail_detection.py

🔮 Future Improvements

Hyperparameter tuning with GridSearchCV

Use deep learning models (RNN, LSTM, Transformers)

Deploy as a Flask/Streamlit web app for real-time email filtering

🙌 Acknowledgements

Scikit-learn

NLTK

SMS Spam Collection Dataset (UCI Repository)

✨ Built with ❤️ to demonstrate text classification with Machine Learning.
