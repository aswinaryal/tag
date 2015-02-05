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
        if k < 400:
            cls = clsX[k]
        else:
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

def tagProbSGDset(tagIdx,setNum=3,typ='cv'):
    opt = 'body+title+code'
    totalKeys = 1000
    testopt = 'l3'
    datastr=typ+str(setNum)
    n=tagIdx.shape[0]
    nkey=tagIdx.shape[1]
    ntask=n*nkey
    tagProb=np.zeros((n, nkey))
    fn = 'data/hvec_'+datastr+'.pkl'
    with open(fn, 'rb') as infile:
        hvec = pickle.load(infile)
    tIdxn1=np.nonzero(tagIdx==-1)
    nt0=len(tIdxn1[0])
    for k in xrange(totalKeys):
        tIdx=np.nonzero(tagIdx==k)
        nt=len(tIdx[0])
        if nt > 0:
            cls = joblib.load('data/SGD_key_'+str(k)+'_'+testopt+'.pkl')
            yp=cls.predict_proba(hvec)
            tagProb[tIdx]=yp[tIdx[0],1]
        nt0 += nt
        if nt0 == ntask:
            break
    if nt0 < ntask:
        #keywords beyond top 1000 keys
        tIdx=np.nonzero(tagProb)
        mp=tagProb[tIdx].mean()
        tIdx=np.nonzero(tagProb==0)
        tagProb[tIdx]=mp
        tagProb[tIdxn1]=0
        print 'mean sgd prob:', mp, ntask-nt0, ntask

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

def initEnvironment():
    nset=30
    a1=np.arange(0,21)*500
    a2=np.arange(1,21)*1000+10000
    a3=np.arange(1,7)*2000+30000
    kk=np.concatenate((a1,a2,a3))
    keyarr=kk[0:(nset+1)]
    global bodyX
    global titleX
    global codeX
    titleX={}
    print 'loading title co-occurrence matrices'
    for i in xrange(0,len(keyarr)-1):
        fn = 'data/titleMatrix_'+str(keyarr[i])+'-'+str(keyarr[i+1])+'_norm.pkl'
        with open(fn, 'rb') as infile:
            x = pickle.load(infile)
        titleX[i] = x.transpose()
    bodyX={}
    print 'loading body co-occurrence matrices'
    for i in xrange(0,len(keyarr)-1):
        fn = 'data/bodyMatrix_'+str(keyarr[i])+'-'+str(keyarr[i+1])+'_norm.pkl'
        with open(fn, 'rb') as infile:
            x = pickle.load(infile)
        bodyX[i] = x.transpose()
    codeX={}
    print 'loading code co-occurrence matrices'
    for i in xrange(0,len(keyarr)-1):
        fn = 'data/codeMatrix_'+str(keyarr[i])+'-'+str(keyarr[i+1])+'_norm.pkl'
        with open(fn, 'rb') as infile:
            x = pickle.load(infile)
        codeX[i] = x.transpose()

    global stopwordX
    stopwordX={}
    print 'loading stopwords'
    for part in ['body','code']:
        with open('data/'+part+'_stopwords_n46t20cv3.pkl','rb') as infile:
            sw = pickle.load(infile)
            stopwordX[part]=set(sw)

    global allKeys
    print 'loading all keywords'
    allKeys=[l.split()[0] for l in open('data/keywordsAll.txt').readlines()]

    global svmX
    print 'loading decision functions'
    svmX = {}
    for opt in ['n46t20cv3body', 'n46t20cv3', 'n46t20cv3sgd', 'n46t20cv3title']:
        svmX[opt] = joblib.load('data/svc_'+opt+'.pkl')

    global clsX
    print 'loading 400 one-vs-all models'
    clsX = {}
    testopt = 'l3'
    for k in range(400):
        clsX[k] = joblib.load('data/SGD_key_'+str(k)+'_'+testopt+'.pkl')

def tagProbability(part,keyn1,keyn2,wvec,ntag,i):
    if i<30:
        if part=='body':
            p = -wvec*bodyX[i]
        elif part=='title':
            p = -wvec*titleX[i]
        else:
            p = -wvec*codeX[i]
    else:
        fn = 'data/'+part+'Matrix_'+str(keyn1)+'-'+str(keyn2)+'_norm.pkl'
        with open(fn, 'rb') as infile:
            x = pickle.load(infile)
        p = -wvec*x.transpose()
    tagIdx=np.zeros((1,ntag),dtype=np.int)
    tagProb=np.zeros((1,ntag))
    prob = p.A.flatten()
    idx = np.argsort(prob)
    tagIdx[0,:]=idx[0:ntag]+keyn1
    tagProb[0,:]=-prob[idx[0:ntag]]
    return tagProb, tagIdx

def combineTag(part,nset,wvec,ntag):
    a1=np.arange(0,21)*500
    a2=np.arange(1,21)*1000+10000
    a3=np.arange(1,7)*2000+30000
    kk=np.concatenate((a1,a2,a3))
    keyarr=kk[0:(nset+1)]
    tagProb, tagIdx = tagProbability(part,keyarr[0],keyarr[1],wvec,ntag,0)
    n=wvec.shape[0]
    for i in xrange(1,len(keyarr)-1):
        tagProb1, tagIdx1 = tagProbability(part,keyarr[i],keyarr[i+1],wvec,ntag,i)
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

def tagAccuracy(tagIdx,xtag):
    ntag=tagIdx.shape[1]
    ntest=tagIdx.shape[0]
    accu=np.zeros((ntest, ntag))
    for i, d in xtag.items():
        for j in xrange(ntag):
            if tagIdx[i,j] in d:
                accu[i,j]=1
    return accu

def singleTagAccuracy(typ='test',setNum=1,nset=46,npred=20):
    parts = ['title','body','code']
    stopword = [None, 'v1', 'v1']
    weight = [1.0, 1.0, 1.0]
    sgdWt = 1.0
    sa=np.zeros(4)
    with open('data/xtag_'+typ+str(setNum)+'.pkl', 'rb') as infile:
        xtag = pickle.load(infile)
    for m, part in enumerate(parts):
        datastr=typ+str(setNum)
        if stopword[m] == 'v1':
            datastr+='sw1'
        opt='n'+str(nset)+'t'+str(npred)+datastr
        fn1 = 'data/'+part+'_tagProb_'+opt+'.pkl'
        fn2 = 'data/'+part+'_tagIdx_'+opt+'.pkl'
        with open(fn1, 'rb') as infile:
            if m==0:
                tagProb1 = pickle.load(infile)
                n=tagProb1.shape[0]
            else:
                tagProb = pickle.load(infile)
        with open(fn2, 'rb') as infile:
            if m==0:
                tagIdx1 = pickle.load(infile)
            else:
                tagIdx = pickle.load(infile)
        if m>0:
            tagProb1 = np.hstack((tagProb1,weight[m]*tagProb))
            tagIdx1 = np.hstack((tagIdx1,tagIdx))
        for j in xrange(n):
            idxDict = defaultdict(int)
            for i in xrange(npred):
                idxDict[tagIdx1[j,i]]=i
            for i in xrange(npred,(m+1)*npred):
                idx=idxDict.get(tagIdx1[j,i])
                if idx is None:
                    if tagIdx1[j,i] != -1:
                        idxDict[tagIdx1[j,i]]=i
                else:
                    tagIdx1[j,i]=-1
                    tagProb1[j,idx]+=tagProb1[j,i]
                    tagProb1[j,i]=-1
            idx = np.argsort(-tagProb1[j,:])
            tagIdx1[j,:]=tagIdx1[j,idx]
            tagProb1[j,:]=tagProb1[j,idx]
        accu = tagAccuracy(np.reshape(tagIdx1[:,0],(n,1)),xtag)
        sa[m] = accu.mean()
        print sa[m]

    tagProbsgd1=tagProbSGDset(tagIdx1,setNum=setNum,typ=typ)
    tagProb1 += sgdWt * tagProbsgd1
    for j in xrange(n):
        idx = np.argsort(-tagProb1[j,:])
        tagIdx[j,:]=tagIdx1[j,idx[0:npred]]
        tagProb[j,:]=tagProb1[j,idx[0:npred]]
    accu = tagAccuracy(np.reshape(tagIdx[:,0],(n,1)),xtag)
    sa[3] = accu.mean()
    print sa[3]
    return sa


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
    for i, part in enumerate(parts):
        if i==0:
            tagProb1, tagIdx1 = tagWordAssociationCV(part,textToken[i],nset=nset,npred=npred,stopword=stopword[i])
            if tit != '':
                tagIdx = tagIdx1
                tagProb = tagProb1
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

    if tit != '':
        tags = []
        for i in range(3):
            tags.append(allKeys[tagIdx[0,i]])
        return tags

    svm = svmX.get('n46t20cv3'+sgd+tit)
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
        if stopwordX.get(part) is not None:
            sw = stopwordX.get(part)
        else:
            with open('data/'+part+'_stopwords_n46t20cv3.pkl','rb') as infile:
                sw = set(pickle.load(infile))
        wvec = getTestVec(part,textToken,stopword=sw)
    else:
        wvec = getTestVec(part,textToken)

    tagProb, tagIdx = combineTag(part,nset,wvec,npred)
    return tagProb, tagIdx


