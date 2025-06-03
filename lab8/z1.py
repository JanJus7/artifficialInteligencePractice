import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('stopwords')

# podpunkt a)
with open('wordsArticle.txt', 'r', encoding='utf-8') as f:
    words = f.read()

tokenizedWords = word_tokenize(words)

# podpunkt b)
print(f"Number of words in the file: {len(tokenizedWords)}")

# podpunkt c)
stopWords = set(stopwords.words('english'))

filteredSentence = []

for w in tokenizedWords:
    if w not in stopWords:
        filteredSentence.append(w)

print(f"Number of words after removing stopwords: {len(filteredSentence)}")

# podpunkt e)
tagged = nltk.pos_tag(filteredSentence)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word.lower(), get_wordnet_pos(pos)) for word, pos in tagged]

print(f"Number of unque words after lematization: {len(set(lemmatized))}")

# podpunkt f)
wordCount = Counter(lemmatized)

mostCommon = wordCount.most_common(10)

words = [word for word, _ in mostCommon]
counts = [count for _, count in mostCommon]

plt.figure(figsize=(10, 6))
plt.bar(words, counts)
plt.xlabel('Words')
plt.ylabel('Ammount')
plt.title('Top 10 most common words.')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# podpunkt g)
text_for_cloud = ' '.join(lemmatized)

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    max_words=100
).generate(text_for_cloud)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()