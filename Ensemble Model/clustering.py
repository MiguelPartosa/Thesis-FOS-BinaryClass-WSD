from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from pprint import pprint
from pathlib import Path
# tokenize, remove stopwords, non-alphabetic words, lowercase
# TODO: Change this to 

# Since Path.cwd() outputs from the parent directory, we have to preappend this and any other changes to path
current_dir = 'Clustering'

def preprocess(textstring):

    stops = set(stopwords.words('english'))
    tokens = word_tokenize(textstring)
    return [token.lower() for token in tokens if token.isalpha()
            and token not in stops]


# This is a very slow process, need to refactor for better visual cues
data_path = Path.joinpath(Path.cwd(), current_dir, 'booksummaries.txt')
print(data_path)
summaries = []
for line in open(data_path, encoding="utf-8"):
    temp = line.split("\t")
    summaries.append(preprocess(temp[6]))

# Create a dictionary representation of the documents.
dictionary = Dictionary(summaries)
# Filter infrequent or too frequent words.
dictionary.filter_extremes(no_below=10, no_above=0.5)
corpus = [dictionary.doc2bow(summary) for summary in summaries]
# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token
# Train the topic model
model = LdaModel(corpus=corpus, id2word=id2word, iterations=400, num_topics=10)
top_topics = list(model.top_topics(corpus))
pprint(top_topics)
