
# # Importing the necessary libraries:



import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# # Loading the 20 Newsgroups dataset for the specified classes:



categories = ['comp.graphics', 'misc.forsale', 'rec.sport.baseball', 'soc.religion.christian', 'talk.politics.guns']
newsgroups_data = fetch_20newsgroups(categories=categories, subset='all', remove=('headers', 'footers', 'quotes'))



newsgroups_data.filenames.shape


# # Get the category labels



labels = newsgroups_data.target


# #  Prepocessing the texts :
# * Transorm to LowerCase
# * Remove Punctuation
# * Remove dates
# * Tokenize
# * Remove StopWords



preprocessed_texts = []
for text in newsgroups_data.data:
    # Lowercase the text
    text = text.lower()
    # Remove punctuations, quotations, and dates
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    # Append the preprocessed text to the list
    preprocessed_texts.append(' '.join(tokens))


# ## Create a DataFrame with labels and preprocessed texts:


data = pd.DataFrame({'labels': labels, 'preprocessed_texts': preprocessed_texts})
data


# # Feature Extraction Based On TF-IDF :

# ## Initialize the TF-IDF vectorizer


vectorizer = TfidfVectorizer()


# ## Fit the vectorizer on the preprocessed texts to learn the vocabulary and IDF values




vectorizer.fit(preprocessed_texts)


# ## Transform the preprocessed texts into TF-IDF feature vectors



features = vectorizer.transform(preprocessed_texts)





print(features.shape)
print(labels.shape)


# # Classification using SVM :

# ## Splitting into Train-Test 


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)


# ## Initialize the Nu-SVC classifier with the specified maximum value of nu (0.5 in this case)



svm = NuSVC(nu=0.5)


# ## Train the SVM classifier 



svm.fit(X_train, y_train)


# ## Make predictions on the testing data



y_pred = svm.predict(X_test)


# ## Calculate precision



precision = precision_score(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.4f}")


# ## Calculate recall



recall = recall_score(y_test, y_pred, average='weighted')
print(f"Recall: {recall:.4f}")


# ## Calculate F1-score



f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-score: {f1:.4f}")


# ## Calculate the classification report



report = classification_report(y_test, y_pred)

print(report)

