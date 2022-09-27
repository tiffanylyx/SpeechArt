import nltk
def preprocessing(corpus):
    clean_text = []
    for row in corpus:
        # tokenize
        tokens = nltk.tokenize.word_tokenize(row)
        # lowercase
        tokens = [token.lower() for token in tokens]
        # isword
        tokens = [token for token in tokens if token.isalpha()]
        clean_sentence = ''
        clean_sentence = ' '.join(token for token in tokens)
        clean_text.append(clean_sentence)
    return clean_text
all_text = ['Florida city gives in to $600,000 bitcoin ransomware demand',
 'Florida city will pay hackers $600,000 to get its computer systems back - The Washington Post',
 'US wants to extradite Swede over $11M Bitcoin investment scam',
 'Bitcoin healthy AF following Zuckbuck reveal, hash rate hits all-time high',
 'Messaging app Line is reportedly launching a cryptocurrency exchange in Japan soon',
 'Facebook’s logo drama is a problem and for more reasons than you think',
 'Here Are All The Deets On Facebook’s New Cryptocurrency, Libra',
 'Facebook’s logo drama is a problem and for more reasons than you think',
 'Florida city agrees to pay hackers $600,000 in bitcoin to get its computer files back',
 'This father of three put everything into bitcoin. Here’s what happened next.',
 "Libra: four reasons to be extremely cautious about Facebook's new currency",
 'What does Facebook’s Libra currency mean for Africa?',
 'With Cryptocurrency Launch, Facebook Sets Its Path Toward Becoming an Independent Nation',
 'What Do Mark Zuckerberg, Marco Rubio, and Patrick Drahi Have in Common?',
 'Florida City Ransom Payment Could Open Door to More Attacks',
 'Facebook cryptocurrency: what it aims to be, why it has led to concern',
 'Post-Ransomware Attack, Florida City Pays $600K',
 'Facebook’s New Libra Digital Currency, Trust Issues (Many), and Sovereignty',
 "Libra: An Interesting Idea, If Only Facebook Weren'tInvolved",
 "Facebook's Libra Needs To Answer Three Questions"]
all_text = preprocessing(all_text)

# sklearn countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
# Convert a collection of text documents to a matrix of token counts
cv = CountVectorizer(stop_words = 'english')
def compute_co_occurrence(sentence_list):


    # matrix of token counts
    X = cv.fit_transform(all_text)

    names = cv.get_feature_names_out()
    edgelist = []
    for index_row, row in enumerate(X.toarray()):
        for index,word in enumerate(row):
            if word>0:
                edgelist.append((names[index_row],names[index],word))
                print(names[index_row],names[index])
    DG = nx.DiGraph()
    DG.add_weighted_edges_from(edgelist)
    l=nx.spring_layout(DG,dim=3,scale = 3)
    return l
l = compute_co_occurrence(all_text)
print(l)
