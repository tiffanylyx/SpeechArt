
import gensim

import gensim.downloader as api

'''

#downloading the Dataset

dataset = api.load("text8")

data = [d for d in dataset]



#creating tagged documents using models.doc2vec.TaggedDcument()

def tagged_doc(list_of_list_of_words):

     for i, list_of_words in enumerate(list_of_list_of_words):

          yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

training_data = list(tagged_doc(data))



#printing the trained dataset

print(training_data[:1])



#initialising the model

dv_model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=30)



#building the vocabulary

dv_model.build_vocab(training_data)



#training the Doc2Vec model

dv_model.train(training_data, total_examples=dv_model.corpus_count, epochs=dv_model.epochs)

dv_model.save("doc2vec100_text8.model")
'''
from gensim.models import Word2Vec
import pickle
from sklearn.decomposition import PCA

dv_model = Word2Vec.load("doc2vec100_text8.model")

#analysing the output
'''
print(dv_model.infer_vector(['describe', 'modern','era','revolution','repudiated']))
sv= []
for i in training_data:
    sv.append(dv_model.infer_vector(i[0]))


pca3 = PCA(n_components=3)
pca3.fit(sv)
with open('pca3_sentenceVec.pkl', 'wb') as pickle_file:
        pickle.dump(pca3, pickle_file)
'''
with open('pca3_sentenceVec.pkl', 'rb') as pickle_file:
    pca3 = pickle.load(pickle_file)

res4 = pca3.transform([dv_model.infer_vector(['describe', 'modern','era','revolution','repudiated'])])
print(res4)
