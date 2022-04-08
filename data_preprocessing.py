import sys
from math import log, sqrt
from itertools import combinations
import json
from nltk.stem.porter import *
from nltk.stem.snowball import ArabicStemmer
from string import punctuation

punctuation += '،؛؟”0123456789“'

stopWords = open(".../arabic_stopwords.txt",encoding = "utf-8").read().splitlines()

words=" ".join(stopWords)
STOPWORDS = frozenset({ w for w in words.split() if w })

def remove_stopword(sentence):
    return " ".join(w for w in sentence.split() if w not in STOPWORDS)

def snowballStemmer(word):
    stemmer = ArabicStemmer()
    stem=stemmer.stem(word)
    return stem


def cosine_distance(a, b):
    cos = 0.0
    a_tfidf = a["tfidf"]
    for token, tfidf in b["tfidf"].items():
        if token in a_tfidf:
            cos += tfidf * a_tfidf[token]
    return cos

def normalize(features):
    if features != {}:
        x=sqrt(sum(i**2 for i in features.values()))
        if x !=0:
            norm = 1.0 / x
            for k, v in features.items():
                features[k] = v * norm
    return features

def add_tfidf_to(documents):
    tokens = {}
    for id, doc in enumerate(documents):
        tf = {}
        doc["tfidf"] = {}
        doc_tokens = doc.get("tokens", [])
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        num_tokens = len(doc_tokens)
        if num_tokens > 0:
            for token, freq in tf.items():
                tokens.setdefault(token, []).append((id, float(freq) / num_tokens))

    doc_count = float(len(documents))
    for token, docs in tokens.items():
        idf = log(doc_count / len(docs))
        for id, tf in docs:
            tfidf = tf * idf
            if tfidf > 0:
                documents[id]["tfidf"][token] = tfidf

    for doc in documents:
        doc["tfidf"] = normalize(doc["tfidf"])

def choose_cluster(node, cluster_lookup, edges):
    new = cluster_lookup[node]
    if node in edges:
        seen, num_seen = {}, {}
        for target, weight in edges.get(node, []):
            seen[cluster_lookup[target]] = seen.get(
                cluster_lookup[target], 0.0) + weight
        for k, v in seen.items():
            num_seen.setdefault(v, []).append(k)
        new = num_seen[max(num_seen)][0]
    return new

def majorclust(graph):
    cluster_lookup = dict((node, i) for i, node in enumerate(graph.nodes))

    count = 0
    movements = set()
    finished = False
    while not finished:
        finished = True
        for node in graph.nodes:
            new = choose_cluster(node, cluster_lookup, graph.edges)
            move = (node, cluster_lookup[node], new)
            if new != cluster_lookup[node] and move not in movements:
                movements.add(move)
                cluster_lookup[node] = new
                finished = False

    clusters = {}
    for k, v in cluster_lookup.items():
        clusters.setdefault(v, []).append(k)

    return clusters.values()

def get_distance_graph(documents):
    class Graph(object):
        def __init__(self):
            self.edges = {}

        def add_edge(self, n1, n2, w):
            if w>=0.0:#mus added            
                self.edges.setdefault(n1, []).append((n2, w))
                self.edges.setdefault(n2, []).append((n1, w))


    graph = Graph()
    doc_ids = range(len(documents))
    graph.nodes = set(doc_ids)
    for a, b in combinations(doc_ids, 2):
        graph.add_edge(a, b, cosine_distance(documents[a], documents[b]))
    return graph

def get_documents(string,ii):
    import sumy
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    LANGUAGE="arabic"
    parser= PlaintextParser(string, Tokenizer(LANGUAGE)).document.sentences

    raw={}
    for s_no, sent in enumerate(parser):
        raw[s_no]=str(sent)

    with open(".../raw"+str(ii)+".json", "w",encoding = "utf-8") as write_file:
        json.dump(raw, write_file)
    write_file.close()

    li=[]
    for i in parser:
         i=str(i)
         line = ''.join(c for c in i if c not in punctuation)
         line= remove_stopword(line)
         line=line.split()
         line=[ snowballStemmer(word) for word in line]#SnowballStemmerx
         li.append(line)
    docs=li
    return [{"s_id":s_id,"text": " ".join(text), "tokens": text} for s_id, text in enumerate(docs)]

'''Main program ...............................'''
def main(args):
    data_x={}
    for ii in range(1,154):
        r_text=open(".../document"+str(ii)+".txt",'r', encoding='utf-8').read()
        documents = get_documents(r_text,ii)
        add_tfidf_to(documents)
        senID={}
        for i in documents:
            senID[i['s_id']]=i['tfidf']
            
        with open(".../tfidf"+str(ii)+".json", "w",encoding = "utf-8") as write_file:
            json.dump(senID, write_file)
        write_file.close()

        dist_graph = get_distance_graph(documents)
        edges=dist_graph.edges

        s={}
        for k,result in list(edges.items()):
            result.sort(key=lambda x: x[1])
            result.reverse()
            dn=[]
            for i in result:
                if i[1] != 0.0:
                    dn.append(i)

            ldn=len(dn)
            
            s[int(k)]=(result,ldn/len(documents))

        with open(".../nearest"+str(ii)+".json", "w",encoding = "utf-8") as write_file:
            json.dump(s, write_file)
        write_file.close()

        print("The end......(",ii,")")   
    

if __name__ == '__main__':
    main(sys.argv)
