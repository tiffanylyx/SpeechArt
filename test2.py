from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader as api
from gensim.models import KeyedVectors
import pickle


model = Word2Vec.load("word2vec_text8.model")
vector = model.wv['computer']  # get numpy vector of a word
word_vectors = KeyedVectors.load("word2vec_text8.wordvectors", mmap='r')
wv = []

for i in range(len(word_vectors)-1):
    wv.append(word_vectors[i])
#word_vectors.save("word2vec_text8.wordvectors")
#wv = KeyedVectors.load("word2vec_text8.wordvectors", mmap='r')

from sklearn.decomposition import PCA

pca4 = PCA(n_components=4)
pca4.fit(wv)
with open('pca4.pkl', 'wb') as pickle_file:
        pickle.dump(pca4, pickle_file)
with open('pca4.pkl', 'rb') as pickle_file:
    pca4 = pickle.load(pickle_file)
res4 = pca4.transform([vector])
print(res4)


pca2 = PCA(n_components=2)
pca2.fit(wv)
with open('pca2.pkl', 'wb') as pickle_file:
        pickle.dump(pca2, pickle_file)
with open('pca2.pkl', 'rb') as pickle_file:
    pca2 = pickle.load(pickle_file)
res2 = pca2.transform([vector])
print(res2)


pca3 = PCA(n_components=3)
pca3.fit(wv)
with open('pca3.pkl', 'wb') as pickle_file:
        pickle.dump(pca3, pickle_file)
with open('pca3.pkl', 'rb') as pickle_file:
    pca3 = pickle.load(pickle_file)
res3 = pca3.transform([vector])
print(res3)
