from collections import defaultdict
import numpy as np
import cPickle as pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

def getTestHvec(textToken):
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 20,
                                   token_pattern=r"[^\s]+",
                                   binary=True, norm=None,
                                   non_negative=True)
    w = ''
    for t in textToken:
        w+=' '+' '.join(t)
    return vectorizer.transform([w])

def tagProbSGD(tagIdx,textToken):
    totalKeys = 1000
    testopt = 'l3'
    nkey=tagIdx.shape[1]
    tagProb=np.zeros((1, nkey))
    hvec = getTestHvec(textToken)   #shape: n x 2^20
    for i in xrange(nkey):
        k = tagIdx[0,i]
        if k < 0 or k >= totalKeys:
            continue
        cls = joblib.load('data/SGD_key_'+str(k)+'_'+testopt+'.pkl')
        yp = cls.predict_proba(hvec)
        tagProb[0,i] = yp[0,1]

    tIdx = np.nonzero(tagProb)
    mp = tagProb[tIdx].mean()
    tIdx = np.nonzero(tagProb==0)
    tagProb[tIdx] = mp
    tIdxn1=np.nonzero(tagIdx==-1)
    tagProb[tIdxn1] = 0
    print 'mean sgd prob:', mp
    return tagProb

def getWords(part,norm=None):
    mincount, maxcount=minWordCount(part)
    words = defaultdict(int)
    with open('data/'+part+'Words.pkl', 'rb') as infile:
        x = pickle.load(infile)
    k=0
    for w in sorted(x, key=x.get, reverse=True):
        if x[w] > mincount and x[w] < maxcount:
            words[w] = k
            if norm is not None:
                norm.append(x[w])
            k+=1
    return words

def getKeys(keyn1,keyn2):
    keys = defaultdict(int)
    p=open('data/keywordsAll.txt').readlines()
    if keyn2==0:  #read all keys
        keyn2=len(p)
    for k, l in enumerate(p[keyn1:keyn2]):
        if k == 0:
            key0 = l.split()[0]
        keys[l.split()[0]] = k
    return keys, key0

def tagProbability(part,keyn1,keyn2,wvec,ntag):
    fn = 'data/'+part+'Matrix_'+str(keyn1)+'-'+str(keyn2)+'_norm.pkl'
    with open(fn, 'rb') as infile:
        x = pickle.load(infile)
    p = -wvec*x.transpose()
    n=wvec.shape[0]
    tagIdx=np.zeros((n,ntag),dtype=np.int)
    tagProb=np.zeros((n,ntag))
    for i in xrange(n):
        prob = p.getrow(i).A.flatten()
        idx = np.argsort(prob)
        tagIdx[i,:]=idx[0:ntag]+keyn1
        tagProb[i,:]=-prob[idx[0:ntag]]
    return tagProb, tagIdx

def combineTag(part,nset,wvec,ntag):
    a1=np.arange(0,21)*500
    a2=np.arange(1,21)*1000+10000
    a3=np.arange(1,7)*2000+30000
    kk=np.concatenate((a1,a2,a3))
    keyarr=kk[0:(nset+1)]
    tagProb, tagIdx = tagProbability(part,keyarr[0],keyarr[1],wvec,ntag)
    n=wvec.shape[0]
    for i in xrange(1,len(keyarr)-1):
        tagProb1, tagIdx1 = tagProbability(part,keyarr[i],keyarr[i+1],wvec,ntag)
        tagProb1 = np.hstack((tagProb,tagProb1))
        tagIdx1 = np.hstack((tagIdx,tagIdx1))
        for j in xrange(n):
            idx = np.argsort(-tagProb1[j,:])
            tagIdx[j,:]=tagIdx1[j,idx[0:ntag]]
            tagProb[j,:]=tagProb1[j,idx[0:ntag]]
    return tagProb, tagIdx


def minWordCount(part):
    minCount=0
    maxCount=1e10
    if part == 'body':
        minCount=1
        maxCount=550000
    elif part == 'code':
        minCount=2
    return minCount, maxCount

def getTestVec(part,textToken,stopword=None):
    words=getWords(part)
    text = []
    vectorizer=TfidfVectorizer(binary=True,vocabulary=words,token_pattern=r"[^\s]+",use_idf=False, norm='l1')
    if stopword==None:
        w=''
        for w1 in textToken:
            if words.get(w1) is not None:
                w+=w1+' '
        text.append(w)
    else:
        w=''
        for w1 in textToken:
            if words.get(w1) is not None:
                if w1 not in stopword:
                    w+=w1+' '
        text.append(w)

    wvec=vectorizer.transform(text)
    return wvec

def tagWordAssociationMultiCV(textToken,nset=40,sgd='',tit=''):
    npred=20
    parts = ['title','body','code']
    stopword = [None, 'v1', 'v1']
    weight = [1.0, 1.0, 1.0]
    sgdWt = 1.0
    n=1
    if tit == 'title':
        parts = ['title']
        npred = 10
    elif tit == 'body':
        parts = ['body']
        stopword = ['v1']
        npred = 10
    print tit
    for i, part in enumerate(parts):
        if i==0:
            tagProb1, tagIdx1 = tagWordAssociationCV(part,textToken[i],nset=nset,npred=npred,stopword=stopword[i])
            if tit != '':
                tagIdx = 1 * tagIdx1
                tagProb = 1.0 * tagProb1
        else:
            tagProb, tagIdx = tagWordAssociationCV(part,textToken[i],nset=nset,npred=npred,stopword=stopword[i])
            tagProb1 = np.hstack((tagProb1,weight[i]*tagProb))
            tagIdx1 = np.hstack((tagIdx1,tagIdx))
    idxDict = defaultdict(int)
    for j in xrange(n):
        for i in xrange(npred):
            idxDict[tagIdx1[j,i]]=i
        for i in xrange(npred,len(parts)*npred):
            idx=idxDict.get(tagIdx1[j,i])
            if idx is None:
                idxDict[tagIdx1[j,i]]=i
            else:
                tagIdx1[j,i]=-1
                tagProb1[j,idx]+=tagProb1[j,i]
                tagProb1[j,i]=-1
    if sgd == 'sgd':
        tagProbsgd1=tagProbSGD(tagIdx1,textToken)
        tagProb1 += sgdWt * tagProbsgd1
    for j in xrange(n):
        idx = np.argsort(-tagProb1[j,:])
        tagIdx[j,:]=tagIdx1[j,idx[0:npred]]
        tagProb[j,:]=tagProb1[j,idx[0:npred]]

    allKeys=[l.split()[0] for l in open('data/keywordsAll.txt').readlines()]

    if tit != '':
        tags = []
        for i in range(3):
            tags.append(allKeys[tagIdx[0,i]])
        return tags

    svm = joblib.load('data/svc_n46t20cv3'+sgd+tit+'.pkl')
    X = np.reshape(tagProb[0,:], (npred, 1))
    y = svm.predict(X)
    tags = [allKeys[tagIdx[0,0]]]
    i=1
    while y[i] > 0:
        tags.append(allKeys[tagIdx[0,i]])
        i+=1
        if i==5:
            break
    return tags


def tagWordAssociationCV(part,textToken,nset=40,npred=20,stopword=None):
    if stopword=='v1':
        with open('data/'+part+'_stopwords_n46t20cv3.pkl','rb') as infile:
            sw = pickle.load(infile)
        wvec = getTestVec(part,textToken,stopword=sw)
    else:
        wvec = getTestVec(part,textToken)

    tagProb, tagIdx = combineTag(part,nset,wvec,npred)
    return tagProb, tagIdx
