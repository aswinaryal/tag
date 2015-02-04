from collections import defaultdict
import json, ijson
import gzip
import sys, os
import numpy as np
import cPickle as pickle
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import random
from trainOneOOCpartial import tagProbSGD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from mean_f1 import mean_f1
import time

def countTotalWords(part):
    x, n0 = resumeCountWord(part)
    for n in xrange(n0,10):
        data = json.load(gzip.open('Train.rdup.'+str(n)+'.json.gz'))
        for i, d in enumerate(data):
            uwords=list(set(d[part]))
            for w in uwords:
                x[w] += 1

        outputName =  part+'Words'+'.'+str(n)+'.pkl'
        with open(outputName, 'wb') as outfile:
            pickle.dump(x, outfile, pickle.HIGHEST_PROTOCOL)
        outputName =  part+'Words'+'.'+str(n-1)+'.pkl'
        if os.path.exists(outputName):
            os.remove(outputName)

    outputName =  part+'Words.pkl'
    with open(outputName, 'wb') as outfile:
        pickle.dump(x, outfile, pickle.HIGHEST_PROTOCOL)

    outputName =  part+'Words.9.pkl'
    if os.path.exists(outputName):
        os.remove(outputName)


def mergeWords():
    wordInTitle = defaultdict(int)
    wordInBody = defaultdict(int)
    wordInCode = defaultdict(int)
    for n in range(1,10):
        data = gzip.open('key.0-500.title.'+str(n)+'.txt.gz').readlines()
        for l in data:
            w, occ = l.split()[0:2]
            wordInTitle[w] += int(occ)
        data = gzip.open('key.0-500.body.'+str(n)+'.txt.gz').readlines()
        for l in data:
            w, occ = l.split()[0:2]
            wordInBody[w] += int(occ)
        data = gzip.open('key.0-500.code.'+str(n)+'.txt.gz').readlines()
        for l in data:
            w, occ = l.split()[0:2]
            wordInCode[w] += int(occ)

    outfile=open('titleWords.txt','w')
    for k, w in enumerate(sorted(wordInTitle, key=wordInTitle.get, reverse=True)):
        outfile.write("%s %d\n" % (w.encode('utf-8'), wordInTitle[w]))
    outfile.close()

    outfile=open('bodyWords.txt','w')
    for k, w in enumerate(sorted(wordInBody, key=wordInBody.get, reverse=True)):
        outfile.write("%s %d\n" % (w.encode('utf-8'), wordInBody[w]))
    outfile.close()

    outfile=open('codeWords.txt','w')
    for k, w in enumerate(sorted(wordInCode, key=wordInCode.get, reverse=True)):
        outfile.write("%s %d\n" % (w.encode('utf-8'), wordInCode[w]))
    outfile.close()

def resumeUsefulWord(part,a0,opt):
    resume = False
    n=0
    for i in xrange(9,-1,-1):
        outputName =  part+'_Dict'+str(a0)+'_'+opt+'.'+str(i)+'.pkl'
        if os.path.exists(outputName):
            if os.path.getsize(outputName)>0:
                with open(outputName, 'rb') as infile:
                    tDict = pickle.load(infile)
                n=i+1
                resume=True
                break
            else:
                os.remove(outputName)
    if not resume:
        tDict = defaultdict(int)
    return tDict, n



def resumeCountWord(part):
    resume = False
    n=1
    for i in xrange(9,0,-1):
        outputName =  part+'Words'+'.'+str(i)+'.pkl'
        if os.path.exists(outputName):
            if os.path.getsize(outputName)>0:
                with open(outputName, 'rb') as infile:
                    x = pickle.load(infile)
                n=i+1
                resume = True
                break
            else:
                os.remove(outputName)
    if not resume:
        x = defaultdict(int)
    return x, n

def resumeCoOccur(part,keyn1,keyn2):
    resume = False
    n=1
    x=0
    for i in xrange(9,0,-1):
        outputName =  part+'Matrix_'+str(keyn1)+'-'+str(keyn2)+'.'+str(i)+'.pkl'
        if os.path.exists(outputName):
            if os.path.getsize(outputName)>0:
                with open(outputName, 'rb') as infile:
                    x = pickle.load(infile)
                n=i+1
                resume = True
                break
            else:
                os.remove(outputName)

    return x, n


def getWords(part,norm=None):
    mincount, maxcount=minWordCount(part)
    words = defaultdict(int)
    with open(part+'Words.pkl', 'rb') as infile:
        x = pickle.load(infile)
    k=0
    for w in sorted(x, key=x.get, reverse=True):
        if x[w] > mincount and x[w] < maxcount:
            words[w] = k
            if norm is not None:
                norm.append(x[w])
            k+=1
    return words

def getWords2(part):
    mincount, maxcount=minWordCount(part)
    with open(part+'Words.pkl', 'rb') as infile:
        x = pickle.load(infile)
    words = [w for w in sorted(x, key=x.get, reverse=True) if x[w] > mincount and x[w] < maxcount]
    return words

def getKeys(keyn1,keyn2):
    keys = defaultdict(int)
    p=open('keywordsAll.txt').readlines()
    if keyn2==0:  #read all keys
        keyn2=len(p)
    for k, l in enumerate(p[keyn1:keyn2]):
        if k == 0:
            key0 = l.split()[0]
        keys[l.split()[0]] = k
    return keys, key0

def coOccurrenceMatrix(part,keyn1,keyn2):
    words=getWords(part)
    keys, key0=getKeys(keyn1,keyn2)
    nkeys=keyn2-keyn1

    x, n0 = resumeCoOccur(part,keyn1,keyn2)
    print 'initial word length:', len(words)
    if n0 == 1:
        x=lil_matrix((nkeys, len(words)))
        #x=lil_matrix((nkeys, len(words)),dtype=np.uint32)
    for n in xrange(n0,10):
        data = json.load(gzip.open('Train.rdup.'+str(n)+'.json.gz'))
        for d in data:
            tidx=[]
            for t in d['tag']:
                idx = keys[t]
                if idx==0 and t != key0:
                    continue
                tidx.append(idx)
            for t in tidx:
                if t == tidx[0]:
                    uwidx=[]
                    for uw in list(set(d[part])):
                        idx = words.get(uw)
                        if idx is not None:
                            uwidx.append(idx)
                for u in uwidx:
                    x[t,u] += 1   #can't change one block all together
        with open(part+'Matrix_'+str(keyn1)+'-'+str(keyn2)+'.'+str(n)+'.pkl', 'wb') as outfile:
            pickle.dump(x, outfile, pickle.HIGHEST_PROTOCOL)
        outputName = part+'Matrix_'+str(keyn1)+'-'+str(keyn2)+'.'+str(n-1)+'.pkl'
        if os.path.exists(outputName):
            os.remove(outputName)
        print 'word length:', len(words), n

    with open(part+'Matrix_'+str(keyn1)+'-'+str(keyn2)+'.pkl', 'wb') as outfile:
        pickle.dump(x.tocsr(), outfile, pickle.HIGHEST_PROTOCOL)

    outputName = part+'Matrix_'+str(keyn1)+'-'+str(keyn2)+'.9.pkl'
    if os.path.exists(outputName):
        os.remove(outputName)


def loadCoMatrix1(fn,norm=None):
    with open(fn, 'rb') as infile:
        x = pickle.load(infile)
        if norm==None:
            return normalize(x, norm='l1', axis=0)
    xs = x.sum(0)              #matrix, should add all the keys
    idx = np.nonzero(xs == 0)
    xs[idx[0]] = 1
    xNorm1 = lil_matrix(1.0/xs)
    return x.multiply(xNorm1[np.arange(1).repeat(x.shape[0]),:])

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

def tagAccuracy(tagIdx,xtag):
    ntag=tagIdx.shape[1]
    ntest=tagIdx.shape[0]
    accu=lil_matrix((ntest, ntag),dtype=np.int)
    for i, d in xtag.items():
        for j in xrange(ntag):
            if tagIdx[i,j] in d:
                accu[i,j]=1
    return accu.tocsr()

def wordUsefulness(part,nset,npred,a0):
    words=getWords2(part)
    setNum=3
    typ='cv'
    datastr=typ+str(setNum)
    opt='n'+str(nset)+'t'+str(npred)+datastr
    fn1='xtag_'+typ+str(setNum)+'.pkl'
    fn2=part+'_wvec_'+datastr+'.pkl'
    with open(fn1, 'rb') as infile:
        xtag = pickle.load(infile)
    with open(fn2, 'rb') as infile:
        wvec = pickle.load(infile)
    with open(part+'_accu_'+opt+'.pkl', 'rb') as infile:
        accu = pickle.load(infile)
    with open(part+'_tagIdx_'+opt+'.pkl', 'rb') as infile:
        tagIdx = pickle.load(infile)
    ntag=tagIdx.shape[1]
    ntest=tagIdx.shape[0]
    A=accu.A
    sa=accu.sum(1).A.flatten()
    stag=np.array([len(d) for i, d in xtag.items()])
    for i in xrange(ntest):
        if sa[i] == stag[i]:
            for j in xrange(ntag-1,-1,-1):
                if A[i,j]:
                    break
                else:
                    A[i,j]=2
    tIdx=np.nonzero(A==a0)
    tTags=tagIdx[tIdx]
    idx=np.argsort(tTags)
    preTag=-1
    print 'dict', len(idx), a0, len(list(set(tTags)))
    tDict, n0 = resumeUsefulWord(part,a0,opt)
    minibatch_size = len(idx)/10 + 1
    j0 = n0*minibatch_size
    nn=np.arange(1,11)*minibatch_size
    nn[-1]=len(idx)
    for j in xrange(j0,len(idx)):
        i = idx[j]
        wv = wvec.getrow(tIdx[0][i]).A.flatten()
        if tTags[i] != preTag:
            xrow = findCoMatrix(part,tTags[i])
            preTag = tTags[i]
        p=wv*xrow
        widx=p.argmax()
        tDict[words[widx]]+=1
        if j== nn[n0]-1:
            fn = part+'_Dict'+str(a0)+'_'+opt+'.'+str(n0)+'.pkl'
            with open(fn, 'wb') as outfile:
                pickle.dump(tDict, outfile, pickle.HIGHEST_PROTOCOL)
            fn = part+'_Dict'+str(a0)+'_'+opt+'.'+str(n0-1)+'.pkl'
            if os.path.exists(fn):
                os.remove(fn)
            n0+=1


    fn = part+'_Dict'+str(a0)+'_'+opt+'.pkl'
    with open(fn, 'wb') as outfile:
        pickle.dump(tDict, outfile, pickle.HIGHEST_PROTOCOL)
    fn = part+'_Dict'+str(a0)+'_'+opt+'.9.pkl'
    if os.path.exists(fn):
        os.remove(fn)

def buildStopWords(part,opt):
    factor = 3
    stopWords=[]
    with open(part+'_Dict0_'+opt+'.pkl', 'rb') as infile:  #misleading words
        dict0 = pickle.load(infile)
    with open(part+'_Dict1_'+opt+'.pkl', 'rb') as infile:  #useful words
        dict1 = pickle.load(infile)
    for w in sorted(dict0, key=dict0.get, reverse=True):
        if w not in dict1:
            stopWords.append(w)
        elif dict0[w] > factor * dict1.get(w):
            stopWords.append(w)
    fn = part+'_stopwords_'+opt+'.pkl'
    with open(fn, 'wb') as outfile:
        pickle.dump(stopWords, outfile, pickle.HIGHEST_PROTOCOL)


def findCoMatrix(part,t):
    a1=np.arange(0,21)*500
    a2=np.arange(1,21)*1000+10000
    a3=np.arange(1,7)*2000+30000
    kk=np.concatenate((a1,a2,a3))
    i=np.nonzero(kk<=t)[0][-1]
    keyn1=kk[i]
    keyn2=kk[i+1]
    fn = part+'Matrix_'+str(keyn1)+'-'+str(keyn2)+'_norm.pkl'
    with open(fn, 'rb') as infile:
        x = pickle.load(infile)
    return x.getrow(t-keyn1).A.flatten()

def loadCoMatrix2(part):
    fn = part+'Matrix_0-500_norm.pkl'
    with open(fn, 'rb') as infile:
        x = pickle.load(infile)
    fn = part+'Matrix_500-1000_norm.pkl'
    with open(fn, 'rb') as infile:
        x2 = pickle.load(infile)
    return x.transpose(), x2.transpose()

def normalizeMatrix(part):
    norm=[]
    words=getWords(part, norm=norm)
    a1=np.arange(0,21)*500
    a2=np.arange(1,21)*1000+10000
    a3=np.arange(1,7)*2000+30000
    kk=np.concatenate((a1,a2,a3))
    xs = np.array(norm)
    idx = np.nonzero(xs)
    for i in xrange(len(kk)-1):
        print kk[i]
        fn = part+'Matrix_'+str(kk[i])+'-'+str(kk[i+1])+'_norm.pkl'
        if os.path.exists(fn):
            if os.path.getsize(fn)>0:
                continue
        if kk[i] == 0:
             row = np.arange(500).repeat(len(idx[0]))
             col = np.tile(idx[0],500)
             data = np.tile(1.0/xs[idx],500)
             xn = csr_matrix((data, (row, col)), shape=(500, len(norm)))
        elif kk[i] == 10000:
             row = np.arange(1000).repeat(len(idx[0]))
             col = np.tile(idx[0],1000)
             data = np.tile(1.0/xs[idx],1000)
             xn = csr_matrix((data, (row, col)), shape=(1000, len(norm)))
        elif kk[i] == 30000:
             row = np.arange(2000).repeat(len(idx[0]))
             col = np.tile(idx[0],2000)
             data = np.tile(1.0/xs[idx],2000)
             xn = csr_matrix((data, (row, col)), shape=(2000, len(norm)))

        fn = part+'Matrix_'+str(kk[i])+'-'+str(kk[i+1])+'.pkl'
        with open(fn, 'rb') as infile:
            x = pickle.load(infile)

        fn = part+'Matrix_'+str(kk[i])+'-'+str(kk[i+1])+'_norm.pkl'
        with open(fn, 'wb') as outfile:
            pickle.dump(x.multiply(xn), outfile, pickle.HIGHEST_PROTOCOL)

def minWordCount(part):
    minCount=0
    maxCount=1e10
    if part == 'body':
        minCount=1
        maxCount=550000
    elif part == 'code':
        minCount=2
    return minCount, maxCount

def getTestVec(part,n,typ,stopword=None, x=None):
    words=getWords(part)
    keys, key0=getKeys(0,0)  #all
    text = []
    k=0
    getTag = False
    if x is not None:
        getTag = True
    vectorizer=TfidfVectorizer(binary=True,vocabulary=words,token_pattern=r"[^\s]+",use_idf=False, norm='l1')
    if stopword==None:
        for d in json.load(gzip.open(typ+str(n)+'.json.gz')):
            w=''
            for w1 in list(set(d[part])):
                if words.get(w1) is not None:
                    w+=w1+' '
            text.append(w)
            if getTag:
                tidx=[]
                for t in d['tag']:
                    tidx.append(keys[t])
                x[k]=tuple(tidx)
            k+=1
    else:
        print 'using stopwords:', stopword[0], stopword[1]
        for d in json.load(gzip.open(typ+str(n)+'.json.gz')):
            w=''
            for w1 in list(set(d[part])):
                if words.get(w1) is not None:
                    if w1 not in stopword:
                        w+=w1+' '
            text.append(w)
            if getTag:
                tidx=[]
                for t in d['tag']:
                    tidx.append(keys[t])
                x[k]=tuple(tidx)
            k+=1

    wvec=vectorizer.transform(text)
    return wvec

def tagWordAssociationMultiCV(nset,npred,calAccu=True,setNum=3,typ='cv',sgd='sgd',tit=''):
    parts = ['title','body','code']
    stopword = [None, 'v1', 'v1']
    weight = [1.0, 1.0, 1.0]
    sgdWt = 1.0
    opt0='n'+str(nset)+'t'+str(npred)+typ+str(setNum)+sgd
    if tit=='title':
        parts = ['title']
        opt0+=tit
    elif tit=='body':
        parts = ['body']
        stopword = ['v1']
        opt0+=tit
    elif tit=='title+body':
        parts = ['title','body']
        stopword = [None, 'v1']

    t1=time.time()
    for i, part in enumerate(parts):
        datastr=typ+str(setNum)
        if stopword[i] == 'v1':
            datastr+='sw1'
        opt='n'+str(nset)+'t'+str(npred)+datastr
        fn1 = part+'_tagProb_'+opt+'.pkl'
        fn2 = part+'_tagIdx_'+opt+'.pkl'
        if not os.path.exists(fn1) or not os.path.exists(fn2):
            tagWordAssociationCV(part,nset,npred,setNum=setNum,typ=typ,stopword=stopword[i],calAccu=False)
        with open(fn1, 'rb') as infile:
            if i==0:
                tagProb1 = pickle.load(infile)
                n=tagProb1.shape[0]
                if tit != '':
                    tagProb = 1.0*tagProb1
            else:
                tagProb = pickle.load(infile)
        with open(fn2, 'rb') as infile:
            if i==0:
                tagIdx1 = pickle.load(infile)
                if tit != '':
                    tagIdx = 1*tagIdx1
            else:
                tagIdx = pickle.load(infile)
        if i>0:
            tagProb1 = np.hstack((tagProb1,weight[i]*tagProb))
            tagIdx1 = np.hstack((tagIdx1,tagIdx))
    t2=time.time()
    for j in xrange(n):
        idxDict = defaultdict(int)
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
    t3=time.time()
    if sgd=='sgd':
        tagProbsgd1=tagProbSGD(tagIdx1,setNum=setNum,typ=typ)
        print 'sgd prob:', tagProbsgd1.min(), tagProbsgd1.max()
        tagProb1 += sgdWt * tagProbsgd1
    t4=time.time()
    for j in xrange(n):
        idx = np.argsort(-tagProb1[j,:])
        tagIdx[j,:]=tagIdx1[j,idx[0:npred]]
        tagProb[j,:]=tagProb1[j,idx[0:npred]]
    t5=time.time()

    if calAccu:
        fn1='xtag_'+typ+str(setNum)+'.pkl'
        with open(fn1, 'rb') as infile:
            xtag = pickle.load(infile)
        accu = tagAccuracy(tagIdx,xtag)
        #with open('all_accu_'+opt0+'.pkl', 'wb') as outfile:
        #    pickle.dump(accu, outfile, pickle.HIGHEST_PROTOCOL)

        #sa=accu.sum(1).A.flatten()
        #stag=np.array([float(len(d)) for i, d in xtag.items()])
        #print 'average score: %.3f' % ((sa/stag).mean())
        sa = accu.A[:,0].mean()
        print 'single tag accuracy: %.3f' % (sa)

    return
        #print 'loading tagProb %.2f s' % (t2-t1)
        #print 'merging tagProb %.2f s' % (t3-t2)
        #print 'adding sgd %.2f s' % (t4-t3)
        #print 'ranking tagProb %.2f s' % (t5-t4)
    with open('all_tagProb_'+opt0+'.pkl', 'wb') as outfile:
        pickle.dump(tagProb, outfile, pickle.HIGHEST_PROTOCOL)
    with open('all_tagIdx_'+opt0+'.pkl', 'wb') as outfile:
        pickle.dump(tagIdx, outfile, pickle.HIGHEST_PROTOCOL)


def findThreshold(nset=46,npred=20,setNum=3,typ='cv',sgd='sgd',tit=''):
    opt0='n'+str(nset)+'t'+str(npred)+typ+str(setNum)+sgd+tit
    with open('all_tagProb_'+opt0+'.pkl', 'rb') as infile:
        tagProb = pickle.load(infile)
    with open('all_accu_'+opt0+'.pkl', 'rb') as infile:
        accu = pickle.load(infile)
    ntest =tagProb.shape[0]
    print 'first pred is right', accu.A[:,0].sum()/float(ntest)
    #tagProbNorm = np.zeros((ntest,npred))
    #for i in xrange(npred):
    #    tagProbNorm[:,i]=tagProb[:,i]/tagProb[:,0]
    svm = LinearSVC(C=1e-1,loss='l1',class_weight={1: 5.0, 0: 1.0})
    n = ntest*npred
    X = np.reshape(tagProb, (n, 1))
    y = np.reshape(accu.A, (n, 1)).flatten()
    svm.fit(X,y)
    ypred = svm.predict(X)
    print 'mean accuracy:', svm.score(X,y)
    print '% of positive:', accu.sum()/float(n)
    print '% of predicted positive', ypred.sum()/float(n)
    filename = 'svc_'+opt0+'.pkl'
    _ = joblib.dump(svm, filename, compress=9)

def finalTagSuggestion(nset=46,npred=20,setNum=3,typ='cv',sgd='sgd',tit=''):
    opt0='n'+str(nset)+'t'+str(npred)+typ+str(setNum)+sgd+tit
    with open('all_tagIdx_'+opt0+'.pkl', 'rb') as infile:
        tagIdx = pickle.load(infile)
    with open('all_tagProb_'+opt0+'.pkl', 'rb') as infile:
        tagProb = pickle.load(infile)
    fn1='xtag_'+typ+str(setNum)+'.pkl'
    with open(fn1, 'rb') as infile:
        xtag = pickle.load(infile)
    svm = joblib.load('svc_n46t20cv3'+sgd+'.pkl')
    n = tagProb.shape[0]
    y_true=[]
    y_pred=[]
    for i, d in xtag.items():
        X = np.reshape(tagProb[i,:], (npred, 1))
        y = svm.predict(X)
        idx=np.nonzero(y)
        npredTags=len(idx[0])
        tags = []
        if npredTags==0:
            tags.append(tagIdx[i,0])
        elif npredTags>5:
            for t in xrange(5):
                tags.append(tagIdx[i,t])
        else:
            for ti in idx[0]:
                tags.append(tagIdx[i,ti])
        y_pred.append(tags)
        y_true.append(list(d))

    print 'mean F1', mean_f1(y_true, y_pred), n
    n2 = int(n*1.08687)
    for i in xrange(n2):
        y_pred.append([1,2,3])
        y_true.append([1,2,3])
    print 'add duplicated'
    print 'mean F1', mean_f1(y_true, y_pred)




def tagWordAssociationCV(part,nset,npred,setNum=3,typ='cv',stopword=None,calAccu=True):
    datastr=typ+str(setNum)
    if stopword=='v1':
        with open(part+'_stopwords_n46t20cv3.pkl','rb') as infile:
            sw = pickle.load(infile)
        datastr+='sw1'

    fn1='xtag_'+typ+str(setNum)+'.pkl'
    fn2=part+'_wvec_'+datastr+'.pkl'
    if os.path.exists(fn1) and os.path.exists(fn2):
        with open(fn1, 'rb') as infile:
            xtag = pickle.load(infile)
        with open(fn2, 'rb') as infile:
            wvec = pickle.load(infile)
    elif os.path.exists(fn1):
        if stopword=='v1':
            wvec = getTestVec(part,setNum,typ,stopword=sw)
        else:
            wvec = getTestVec(part,setNum,typ)
        with open(fn2, 'wb') as outfile:
            pickle.dump(wvec, outfile, pickle.HIGHEST_PROTOCOL)
        with open(fn1, 'rb') as infile:
            xtag = pickle.load(infile)
    else:
        xtag = {}
        if stopword=='v1':
            wvec = getTestVec(part,setNum,typ,stopword=sw,x=xtag)
        else:
            wvec = getTestVec(part,setNum,typ,x=xtag)
        with open(fn1, 'wb') as outfile:
            pickle.dump(xtag, outfile, pickle.HIGHEST_PROTOCOL)
        with open(fn2, 'wb') as outfile:
            pickle.dump(wvec, outfile, pickle.HIGHEST_PROTOCOL)

    opt='n'+str(nset)+'t'+str(npred)+datastr
    tagProb, tagIdx = combineTag(part,nset,wvec,npred)

    with open(part+'_tagProb_'+opt+'.pkl', 'wb') as outfile:
        pickle.dump(tagProb, outfile, pickle.HIGHEST_PROTOCOL)
    with open(part+'_tagIdx_'+opt+'.pkl', 'wb') as outfile:
        pickle.dump(tagIdx, outfile, pickle.HIGHEST_PROTOCOL)
    if calAccu:
        accu = tagAccuracy(tagIdx,xtag)
        with open(part+'_accu_'+opt+'.pkl', 'wb') as outfile:
            pickle.dump(accu, outfile, pickle.HIGHEST_PROTOCOL)

        sa=accu.sum(1).A.flatten()
        stag=np.array([float(len(d)) for i, d in xtag.items()])
        print 'average score: ', (sa/stag).mean()


if __name__ == '__main__':
    #mergeWords()
    #coOccurrenceMatrix(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
    #countTotalWords(sys.argv[1])
    #tagWordAssociationCV(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),setNum=1,typ='test',stopword='v1')
    #tagWordAssociationCV(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),stopword='v1')
    #normalizeMatrix(sys.argv[1])
    #wordUsefulness(sys.argv[1],46,20,int(sys.argv[2]))
    #buildStopWords(sys.argv[1],'n46t20cv3')
    #tagWordAssociationMultiCV(46,20,setNum=3,typ='cv',sgd='',tit='title')
    #tagWordAssociationMultiCV(46,20,setNum=3,typ='cv',sgd='',tit='title+body')
    #tagWordAssociationMultiCV(46,20,setNum=3,typ='cv',sgd='',tit='')
    #tagWordAssociationMultiCV(46,20,setNum=3,typ='cv',sgd='sgd',tit='')
    #tagWordAssociationMultiCV(46,20,setNum=1,typ='test',sgd='sgd')
    #tagWordAssociationMultiCV(46,20,setNum=2,typ='test',sgd='sgd')
    #findThreshold(sgd='',tit='body')
    #finalTagSuggestion(setNum=3,typ='cv',sgd='',tit='body')
    #finalTagSuggestion(setNum=1,typ='test')
    finalTagSuggestion(setNum=2,typ='test')
