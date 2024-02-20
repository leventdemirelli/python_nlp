
#pip install nltk
#pip install textblob
#pip install wordcloud



from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# ONISLEME

df = pd.read_excel("//Users/leventdemirelli/Desktop/SVS.xlsx")

df = df[df['soyleyen'] == 'Recep Tayyip Erdoğan']


df.head()

df["reviewText"]

df["reviewText"] = df["reviewText"].str.lower()

df["reviewText"] = df["reviewText"].str.replace('\d', '')

df["reviewText"] = df["reviewText"].str.replace('[^\w\s]', '')

print(df["reviewText"])

# STOPWORDS

import nltk
nltk.download('stopwords')


sw = stopwords.words('turkish')

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

print(df["reviewText"])





# RAREWORDS

temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()
print(temp_df)

drops = temp_df[temp_df <= 1]

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))



# TOKENIZATION

#nltk.download("punkt")

df["reviewText"].apply(lambda x: TextBlob(x).words).head()



# LEMMATIZATION

#nltk.download('wordnet')

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# FREQUENCY

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

# tf.to_excel('output.xlsx', index=False)

# BARPLOT

tf[tf["tf"] > 15].plot.bar(x="words", y="tf")
plt.show()


# WORDCLOUD

text = " ".join(i for i in tf.words)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")

# Sentiment Analysis

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()

sia.polarity_scores("I liked this music but it is not good as the other one")


df['reviewText'][0:10].apply(lambda x: sia.polarity_scores(x))

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["polarity_score"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])


df = df["reviewText"]
df = str(df)
TextBlob(df).ngrams(2)


from sklearn.feature_extraction.text import CountVectorizer

X = df["reviewText"]

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X_n = vectorizer2.fit_transform(X)

print(X_n)
vectorizer2.get_feature_names_out()
X_n.toarray()


import pandas as pd
import itertools
from collections import Counter

from itertools import combinations

combs = df['reviewText'].apply(lambda x:list(combinations(sorted(x.split()),2)))
counts = Counter(combs.explode())
res = pd.Series(counts).rename_axis(['name1', 'name2']).reset_index(name='count')
print(res)



from collections import Counter
text = df["reviewText"]
text = str(text)
words = text.split()

d = {' '.join(words):n for words,n in Counter(zip(words, words[1:])).items() if not  words[0][-1]==(',')}
print (d)

import json
print (json.dumps(d, indent=4))


import pandas as pd


----


df = pd.read_excel("//Users/leventdemirelli/Desktop/SVS_1.xlsx")

df['notes']

df['notes'] = df['notes'].str.lower()

df['notes'] = df['notes'].str.replace('\d', '')

df['notes'] = df['notes'].str.replace(r'[^\w\s]', '')

df['notes'] = df['notes'].str.replace(':', '')

df['notes'] = df['notes'].str.replace('"', '')

df['notes'] = df['notes'].str.replace("'", '')

df['notes'] = df['notes'].str.replace(u"\u2018", '')

df['notes'] = df['notes'].str.replace(u"\u2019", '')

df['notes'] = df['notes'].str.replace('“', '')

df['notes'] = df['notes'].str.replace(',', '')

df['notes'] = df['notes'].str.replace('.', '')

df['notes'] = df['notes'].str.replace('-', '')

df['notes'] = df['notes'].str.replace('?', '')

df['notes'] = df['notes'].str.replace('”', '')

print(df['notes'])


# STOPWORDS

import nltk
nltk.download('stopwords')


sw = stopwords.words('turkish')

df['notes'] = df['notes'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

print(df['notes'])

# TOKENIZATION

nltk.download("punkt")

df['notes'].apply(lambda x: TextBlob(x).words).head()


# LEMMATIZATION

nltk.download('wordnet')

df['notes'] = df['notes'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))



# RAREWORDS

temp_df = pd.Series(' '.join(df['notes']).split()).value_counts()
print(temp_df)

drops = temp_df[temp_df <= 1]

df['notes'] = df['notes'].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))







df = pd.DataFrame(df['notes'])

from nltk import everygrams, word_tokenize

x = df['notes'].apply(lambda x: [' '.join(ng) for ng in everygrams(word_tokenize(x),2, 2)]).to_frame()

import numpy as np



flattenList = pd.Series(np.concatenate(x.notes))
freqDf = flattenList.value_counts().sort_index().rename_axis('notes').reset_index(name = 'frequency')

freqDf.to_excel("excel_name.xlsx",sheet_name="sheet_name",index=False)


text = " ".join(i for i in freqDf['notes'])

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

import matplotlib.pyplot as plt

plt.figure()
flattenList.value_counts().plot(kind = 'bar', title = 'Count of 1-word and 2-word frequencies')
plt.xlabel('Words')
plt.ylabel('Count')
plt.show()




