from textblob import TextBlob
import pandas as pd
import numpy as np

# train = pd.read_csv('posts.csv', sep=";")
train=pd.read_csv("D://tech_title.csv",encoding='unicode_escape')
# Word Count
print("Word Count")
train['word_count'] = train['content'].apply(lambda x: len(str(x).split(" ")))
print(train[['content', 'word_count']].head())
print("--"*100)

# Char Count
print("Char Count")
train['char_count'] = train['content'].str.len()  ## this also includes spaces
print(train[['content', 'char_count']].head())
print("--"*100)

print("Average Word Count")
def avg_word(sentence):
    words = sentence.split()
    return sum(len(word) for word in words) / len(words)

train['avg_word'] = train['content'].astype(str).apply(lambda x: avg_word(x))
print(train[['content', 'avg_word']].head())
print("--"*100)

print("Stopwords")
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
stop = stopwords.words('english')
train['stopwords'] = train['content'].astype(str).apply(lambda x: len([x for x in x.split() if x in stop]))
print(train[['content', 'stopwords']].head())
print("--"*100)


print("Special Characters")
train['hastags'] = train['content'].astype(str).apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
print(train[['content','hastags']].head())
print("--"*100)

print("Numerics")
train['numerics'] = train['content'].astype(str).apply(lambda x: len([x for x in x.split() if x.isdigit()]))
print(train[['content','numerics']].head())
print("--"*100)

train['content'] = train['content'].astype(str).apply(lambda x: " ".join(x.lower() for x in x.split()))
print(train['content'].head())
print("--"*100)

train['content'] = train['content'].str.replace('[^\w\s]','')
print(train['content'].head())
print("--"*100)

from nltk.corpus import stopwords
stop = stopwords.words('english')
train['content'] = train['content'].astype(str).apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(train['content'].head())
print("--"*100)


freq = pd.Series(' '.join(train['content']).split()).value_counts()[:10]
freq = list(freq.index)
train['content'] = train['content'].astype(str).apply(lambda x: " ".join(x for x in x.split() if x not in freq))
print(train['content'].head())
print("--"*100)

freq = pd.Series(' '.join(train['content']).split()).value_counts()[-10:]
freq = list(freq.index)
train['content'] = train['content'].astype(str).apply(lambda x: " ".join(x for x in x.split() if x not in freq))
print(train['content'].head())
print("--"*100)


print(train['content'][:5].astype(str).apply(lambda x: str(TextBlob(x).correct())))
print("--"*100)


print("Toeknization")
print(TextBlob(train['content'][1]).words)
print("--"*100)

print("Stemming")
from nltk.stem import PorterStemmer
st = PorterStemmer()
print(train['content'][:5].astype(str).apply(lambda x: " ".join([st.stem(word) for word in x.split()])))
print("--"*100)

print("Lemmatization")
from textblob import Word
train['content'] = train['content'].astype(str).apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
print(train['content'].head())
print("--"*100)

# -----------------------Advance Preprocessing---------------------------
print("N-grams")
print(TextBlob(train['content'][0]).ngrams(2))
print("--"*100)

print("Term frequency")
tf1 = (train['content'][1:2]).astype(str).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
print(tf1)
print("--"*100)

for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['content'].str.contains(word)])))
print(tf1)
print("--"*100)

tf1['tfidf'] = tf1['tf'] * tf1['idf']

print(tf1)
print("--"*100)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(train['content'])
print(train_vect)
print("--"*100)

print("Bag of words")
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(train['content'])
print(train_bow)


print("--"*100)
print("Sentiment Analysis")
print(train['content'][:5].astype(str).apply(lambda x: TextBlob(x).sentiment))



