import gzip, json
import sys, os, glob
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.externals import joblib
import cPickle as pickle
import random
import numpy as np

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def normOpt(n):
    if n in ['l1','l2']:
        return n
    return None

def getMiniBatch(data,n1,n2,key,opt):
    body = []
    y = []
    for d in data[n1:n2]:
        w = ' '.join(d['body'])
        if opt != 'body' :
            w += ' ' + ' '.join(d['title'])
            if opt != 'body+title':
                w += ' ' + ' '.join(d['code'])
        body.append(w)
        if key in d['tag']:
            y.append(1)
        else:
            y.append(0)
    return body, y

def getTestSet(n,key,opt,testSize=0.01,seed=123):
    random.seed(seed)
    body = []
    y = []
    for d in json.load(gzip.open('Train.rdup.'+str(n)+'.json.gz')):
        if random.random() <= testSize:
            w = ' '.join(d['body'])
            if opt != 'body' :
                w += ' ' + ' '.join(d['title'])
                if opt != 'body+title':
                    w += ' ' + ' '.join(d['code'])
            body.append(w)
            if key == 'None':
                d['tag'].sort()
                w = ' '.join(d['tag'])
                y.append(w)
            else:
                if key in d['tag']:
                    y.append(1)
                else:
                    y.append(0)
    return body, y

def getTestHvec(n,typ):
    bv = 'True'
    nneg = 'True'
    nv = 'None'
    body = []
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 20,
                                   token_pattern=r"\b\w[\w#+.-]*(?<!\.$)",
                                   binary=str2bool(bv), norm=normOpt(nv),
                                   non_negative=str2bool(nneg))
    for d in json.load(gzip.open(typ+str(n)+'.json.gz')):
        w = ' '.join(d['body'])
        w += ' ' + ' '.join(d['title'])
        w += ' ' + ' '.join(d['code'])
        body.append(w)
    wvec=vectorizer.transform(body)
    return wvec

def resumeJob(outputName, pklName):
    resume = False
    n=1
    ntrain=0
    if os.path.exists(outputName) and os.path.exists(pklName):
        if os.path.getsize(outputName)>0 and os.path.getsize(pklName)>0:
            p=open(outputName).readlines()
            n=int(p[-1].split()[0])+1
            ntrain=int(p[-1].split()[1])
            if n>=2:
                resume = True
    if not resume:
        try:
            os.remove(outputName)
            os.remove(pklName)
        except OSError:
            pass
    return n, ntrain


def run(keyn, nPart):
    all_classes = np.array([0, 1])
    allKeys=[l.split()[0] for l in open('keywordsAll.txt').readlines()]
    keyFreqs=[float(l.split()[1])/4205907 for l in open('keywordsAll.txt').readlines()]
    key = allKeys[keyn]
    freq = keyFreqs[keyn]

    opt = 'body+title+code'
    bv = 'True'
    nneg = 'True'
    nv = 'None'
    #testopt = 'c'
    #testopt = 'w'
    #testopt = 'l2'
    testopt = 'l1'

    if testopt == 'c':
        cls = SGDClassifier(loss='hinge',learning_rate="constant",alpha=1e-6,eta0=1e-2,penalty='l2')
    elif testopt == 'w':
        cls = SGDClassifier(class_weight={1: 1.0/freq/8.0, 0: 1})
    elif testopt == 'l2':
        cls = SGDClassifier(loss='log',alpha=1e-5,penalty='l2')
    elif testopt == 'l1':
        cls = SGDClassifier(loss='log',alpha=1e-5,penalty='l1')

    outputName = 'key_'+str(keyn)+'_SGDtune_'+opt+'_partialfit_'+testopt+'.txt'
    pklName = 'SGD_key_'+str(keyn)+'_'+testopt+'.pkl'
    n0, ntrain = resumeJob(outputName, pklName)

    body_test, y_test = getTestSet(10,key,opt,testSize=0.2,seed=123)
    tot_pos = sum(y_test)
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 20,
                                   token_pattern=r"\b\w[\w#+.-]*(?<!\.$)",
                                   binary=str2bool(bv), norm=normOpt(nv),
                                   non_negative=str2bool(nneg))

    X_test = vectorizer.transform(body_test)
    #print 'test case:', len(y_test), 'positive', tot_pos, 'key:', key, 'X norm:', X_test.sum(), 'binary:', bv, 'norm:', nv, 'nneg:', nneg
    if n0 >= 2:
        cls = joblib.load(pklName)
    for n in xrange(n0,10):
        outfile=open(outputName,'a')
        data = json.load(gzip.open('Train.rdup.'+str(n)+'.json.gz'))
        minibatch_size = len(data)/nPart + 1
        for i in xrange(nPart):
            n1 = i*minibatch_size
            n2 = (i+1)*minibatch_size
            if i==nPart-1:
                n2=len(data)
            ntrain += (n2 - n1)
            body_train, y_train = getMiniBatch(data,n1,n2,key,opt)
            X_train = vectorizer.transform(body_train)
            shuffledRange = range(n2-n1)
            for n_iter in xrange(5):
                X_train, y_train = shuffle(X_train, y_train)
            cls.partial_fit(X_train, y_train, classes=all_classes)
            y_pred = cls.predict(X_test)
            f1 = metrics.f1_score(y_test, y_pred)
            p = metrics.precision_score(y_test, y_pred)
            r = metrics.recall_score(y_test, y_pred)
            accu = cls.score(X_train, y_train)
            y_pred = cls.predict(X_train)
            f1t = metrics.f1_score(y_train, y_pred)
            outfile.write("%3d %8d %.4f %.3f %.3f %.3f %.3f %5d  %5d\n" % (n, ntrain, accu, f1t, f1, p, r, sum(y_pred), tot_pos))
        _ = joblib.dump(cls, pklName, compress=9)
        outfile.close()

def tagProbSGD(tagIdx,setNum=3,typ='cv'):
    opt = 'body+title+code'
    totalKeys = 1000
    testopt = 'l3'
    datastr=typ+str(setNum)
    n=tagIdx.shape[0]
    nkey=tagIdx.shape[1]
    ntask=n*nkey
    tagProb=np.zeros((n, nkey))
    fn = 'hvec_'+datastr+'.pkl'
    if os.path.exists(fn):
        with open(fn, 'rb') as infile:
            hvec = pickle.load(infile)
    else:
        hvec = getTestHvec(setNum,typ)   #shape: n x 2^20
        with open(fn, 'wb') as outfile:
            pickle.dump(hvec, outfile, pickle.HIGHEST_PROTOCOL)
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

def predictTag():
    all_classes = np.array([0, 1])
    allKeys=[l.split()[0] for l in open('keywordsAll.txt').readlines()]
    opt = 'body+title+code'
    bv = 'True'
    nneg = 'True'
    nv = 'None'
    testopt = 'l2'
    #keyNum = [int(f.split('_')[2]) for f in glob.glob('SGD_key_*_'+testopt+'.pkl')]
    keyNum = [i for i in range(1000)]
    nkey = len(keyNum)
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 20,
                                   token_pattern=r"\b\w[\w#+.-]*(?<!\.$)",
                                   binary=str2bool(bv), norm=normOpt(nv),
                                   non_negative=str2bool(nneg))
    body_test, y_test = getTestSet(10,'None',opt,testSize=0.2,seed=123)
    X_test = vectorizer.transform(body_test)
    tagProb=np.zeros((len(y_test), nkey))
    for k, keyn in enumerate(keyNum):
        cls = joblib.load('SGD_key_'+str(keyn)+'_'+testopt+'.pkl')
        yp=cls.predict_proba(X_test)
        tagProb[:,k]=yp[:,1]
    score=np.zeros(len(y_test))
    minProb=np.zeros(len(y_test))
    ntagAboveMinProb=np.zeros(len(y_test))
    for i, y in enumerate(y_test):
        tags = y.split()
        idx = np.argsort(-tagProb[i,:])
        minp = 1e6
        tagPos = 0
        s=0.0
        for j in range(5):
            keyn = keyNum[idx[j]]
            if allKeys[keyn] in tags:
                s+=1
                if tagProb[i,idx[j]] < minp:
                    minp = tagProb[i,idx[j]]
                    tagPos = j
        score[i] = s/len(tags)
        minProb[i] = minp
        ntagAboveMinProb[i] = tagPos
    print 'average score: ', score.mean()
    idx = np.nonzero(score)
    print len(idx), len(idx[0]), len(y_test)
    print 'min prob: ', minProb[idx].mean()
    print 'ntag above: ', ntagAboveMinProb[idx].mean()


if __name__ == '__main__':
    #run(int(sys.argv[1]), 10)
    predictTag()


