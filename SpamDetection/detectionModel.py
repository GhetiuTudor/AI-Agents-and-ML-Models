import requests 
import zipfile
import io
import os
import pandas 
import nltk 
import re
import sys
#from nltk.tokenize import word_tokenize
 
 #download & extract training data (SMS Messages)
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
response = requests.get(url)
if response.status_code == 200: 
    print("Download ok")
else: 
    print("Download failed")

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("sms_spam_collection")
    print("Extraction ok")

extracted = os.listdir("sms_spam_collection")
print("Extracted files: ", extracted)

df = pandas.read_csv(
    "sms_spam_collection/SMSSpamCollection",
    sep="\t", #tab separated 
    header=None, 
    names=["label", "message"],
)

#display the messages
print("_head_")
print(df.head()) #ham for benign messages
print("_description_")
print(df.describe())
print("_info_")
print(df.info())

print("missing values: \n", df.isnull().sum())

print("duplicates: ", df.duplicated().sum())
df = df.drop_duplicates()

#preprocessing the spam dataset 
nltk.download("punkt")
#nltk.download("punkt_tab")
nltk.download("stopwords")

print("before processing")
print(df.head(5))

#lowercasing to standardise emphasis 
df["message"] = df["message"].str.lower()
print("\nafter lowercasing")
print(df["message"].head(5))

#remove unnecessary punctuation -> everything beside $ and ! 
df["message"] = df["message"].apply(lambda x: re.sub(r"[^a-z\s$!]", "", x))
print("\npunctuation removal ")
print(df["message"].head(5))

#tokenisation 
from nltk.tokenize import word_tokenize
df["message"] = df["message"].apply(word_tokenize)
print("\nafter tokenization ")
print(df["message"].head(5))

from nltk.corpus import stopwords

# remove stop words 
stop_words = set(stopwords.words("english"))
df["message"] = df["message"].apply(lambda x: [word for word in x if word not in stop_words])
print("\nremoving stop words")
print(df["message"].head(5))

from nltk.stem import PorterStemmer

# stemming to get words back to their initial form 
stemmer = PorterStemmer()
df["message"] = df["message"].apply(lambda x: [stemmer.stem(word) for word in x])
print("\nstemming ")
print(df["message"].head(5))

#rejoin tokens for feature extraction 
df["message"] = df["message"].apply(lambda x: " ".join(x))
print("\njoining tokens back into strings ", flush=True)
print(df["message"].head(5), flush= True)

#feature extraction y 
#at this step training data looks like a matrix of frequences 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df = 1, max_df= 0.9, ngram_range=(1,2))
X = vectorizer.fit_transform(df["message"])
y = df["label"].apply(lambda x: 1 if x== "spam" else 0)


#training for spam detection (multinomial naive bayes)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

#the pipeline for streamlining the process
pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", MultinomialNB())
])

#tunning
param_grid = {
    "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1"
)

grid_search.fit(df["message"], y)


best_model = grid_search.best_estimator_
print("best model parameters :", grid_search.best_params_)


#evaluation 
new_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still meeting up for lunch today?",
    "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
    "Reminder: Your appointment is scheduled for tomorrow at 10am.",
    "FREE entry in a weekly competition to win an iPad. Just text WIN to 80085 now!",
]

import numpy as np

# Preprocess function that mirrors the training-time preprocessing
def preprocess_message(message):
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

processed = [preprocess_message(msg) for msg in new_messages]

X_new = best_model.named_steps["vectorizer"].transform(processed)

predictions = best_model.named_steps["classifier"].predict(X_new)
prediction_prob= best_model.named_steps["classifier"].predict_proba(X_new)

for i, msg in enumerate(new_messages):
    prediction = "Spam" if predictions[i] == 1 else "Not-Spam"
    spam_probability = prediction_prob[i][1]  # Probability of being spam
    ham_probability = prediction_prob[i][0]   # Probability of being not spam
    
    print(f"Message: {msg}")
    print(f"Prediction: {prediction}")
    print(f"Spam Probability: {spam_probability:.2f}")
    print(f"Not-Spam Probability: {ham_probability:.2f}")
    print("-" * 50)

import joblib
model = 'spam_detection_model.joblib'
joblib.dump(best_model, model)
print(f"Model saved to {model}")
