import sys
from math import log, sqrt
from itertools import combinations
import json
import en_core_web_sm
nlp = en_core_web_sm.load()

for ii in range(1,11):
    with open(".../raw"+str(ii)+".json", "r") as read_file:
        data = json.load(read_file)
    read_file.close()

    with open(".../tfidf"+str(ii)+".json", "r") as read_file:
        tfidf = json.load(read_file)
    read_file.close()

    with open("D.../nearest"+str(ii)+".json", "r") as read_file:
        nearest = json.load(read_file)
    read_file.close()

    cc=[]
    for v in  data.values():
        cc.append(len(list(v.split())))
    maxLen=max(cc)

    f={}
    ne=0
    for k,v in  data.items():
        sumTFIDF=sum(list(tfidf[k].values()))/len(list(v.split()))
        pos=max((1/(int(k)+1)),(1/(len(list(data.keys()))-(int(k)+1)+1)))
        length=len(list(v.split()))/maxLen
        r=[(X.text, X.label_) for X in nlp(v).ents]
        if r !=[]:
            ne=len(r)/len(list(v.split()))
        else:
            ne=0
        f[k]=(v,(pos,sumTFIDF,length,ne,nearest[k][1]))
    with open(".../features"+str(ii)+".json", "w") as write_file:
        json.dump(f, write_file)
    write_file.close()


