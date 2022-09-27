
from nltk.corpus import cmudict

def compute_syllables(word):
    d = cmudict.dict()
    for x in d[word.lower()]:\
        count = 0
        list = []
        for y in x:
            if y[-1].isdigit():
                list.append(count)
                count+=1
            else:
                count+=len(y)
    return list
