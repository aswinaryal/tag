import gzip, json
import sys
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.externals import joblib
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
            if key in d['tag']:
                y.append(1)
            else:
                y.append(0)
    return body, y


def run(keyn, nPart):
    all_classes = np.array([0, 1])
    allKeys=[l.split()[0] for l in open('keywordsAll.txt').readlines()]
    keyFreqs=[float(l.split()[1])/4205907 for l in open('keywordsAll.txt').readlines()]
    key = allKeys[keyn]
    freq = keyFreqs[keyn]

    options = ['body+title+code']
    binaryVec = ['True']
    nnegVec = ['True']
    normVec = ['None']
    
    # p:n = 1:6 to 1:10 seem all good
    #testOpt = 'cw' # class weight
    #partial_fit_classifiers = {
    #    'SGD-org': SGDClassifier(),
    #    'SGD-org-cw1': SGDClassifier(class_weight={1: 1.0/freq/4.0, 0: 1}),
    #    'SGD-org-cw2': SGDClassifier(class_weight={1: 1.0/freq/6.0, 0: 1}),
    #    'SGD-org-cw3': SGDClassifier(class_weight={1: 1.0/freq/8.0, 0: 1}),
    #    'SGD-org-cw4': SGDClassifier(class_weight={1: 1.0/freq/10.0, 0: 1})
    #}
    testOpt = 'lp' #loss, penalty
    partial_fit_classifiers = {
        'SGD': SGDClassifier()
    }
    for l0 in ['hinge','log','modified_huber']:
        for p0 in ['l2', 'l1']:
            for a0 in [1e-8,1e-6,1e-4,1e-2]:
                for pw in [0.3,0.5,0.7]:
                    tag = 'SGD-' + l0 + '-' + p0 + '-a'+str(int(np.log10(a0)))+'-p'+str(pw)
                    partial_fit_classifiers[tag] = SGDClassifier(loss=l0,alpha=a0,power_t=pw,penalty=p0)

    for opt in options:
        outfile=open('key_'+str(keyn)+'_SGDtune_'+opt+'_partialfit_'+testOpt+'.txt','w')
        body_test, y_test = getTestSet(10,key,opt,testSize=0.05,seed=123)
        tot_pos = sum(y_test)
        for nv in normVec:
            for bv in binaryVec:
                for nneg in nnegVec:
                    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 20,
                                                   token_pattern=r"\b\w[\w#+.-]*(?<!\.$)",
                                                   binary=str2bool(bv), norm=normOpt(nv),
                                                   non_negative=str2bool(nneg))

                    X_test = vectorizer.transform(body_test)
                    #print 'test case:', len(y_test), 'positive', tot_pos, 'key:', key, 'X norm:', X_test.sum(), 'binary:', bv, 'norm:', nv, 'nneg:', nneg
                    ntrain = 0
                    for n in xrange(1,4):
                        data = json.load(gzip.open('Train.rdup.'+str(n)+'.json.gz'))
                        minibatch_size = len(data)/nPart + 1
                        #print 'batch size:', minibatch_size
                        for i in xrange(nPart):
                            n1 = i*minibatch_size
                            n2 = (i+1)*minibatch_size
                            if i==nPart-1:
                                n2=len(data)
                            ntrain += (n2 - n1)
                            body_train, y_train = getMiniBatch(data,n1,n2,key,opt)
                            X_train = vectorizer.transform(body_train)
                            for cls_name, cls in partial_fit_classifiers.items():
                                #shuffledRange = range(n2-n1)
                                #for n_iter in xrange(5):
                                #    X_train, y_train = shuffle(X_train, y_train)
                                cls.partial_fit(X_train, y_train, classes=all_classes)
                                y_pred = cls.predict(X_test)
                                f1 = metrics.f1_score(y_test, y_pred)
                                p = metrics.precision_score(y_test, y_pred)
                                r = metrics.recall_score(y_test, y_pred)
                                accu = cls.score(X_train, y_train)
                                y_pred = cls.predict(X_train)
                                f1t = metrics.f1_score(y_train, y_pred)
                                #scores = cross_validation.cross_val_score(cls, X2_train, y_train, cv=3, scoring='f1')
                                #print cls_name, scores
                                outfile.write("%22s %8d %.4f %.3f %.3f %.3f %.3f %5d  %5d\n" % (cls_name, ntrain, accu, f1t, f1, p, r, sum(y_pred), tot_pos))
        outfile.close()






run(int(sys.argv[1]), 10)

