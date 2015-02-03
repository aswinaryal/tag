from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import socket
import requests
import numpy as np
from cleanData import body2Token, title2Token
from coOccurrence1 import tagWordAssociationMultiCV

#######################
#### configuration ####
#######################

app = Flask(__name__)

################
#### routes ####
################


@app.route('/notebook')
def notebook():
    return render_template('test.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = []
    texterr = False
    urlerr = False
    raw = ''
    url = ''
    answers = ''
    if request.method == "POST":
        # get text that the person has entered
        try:
            raw = request.form['text']
        except:
            texterr = True
            pass
        try:
            url = request.form['url']
        except:
            urlerr = True
            pass

        if texterr and urlerr:
            errors.append(
                "Unable to get text. Please enter again."
            )
            return render_template('index.html', errors=errors)
        if raw or url:
            if url:
                r = requests.get(url)
                soup = BeautifulSoup(r.text)
                trueTags = [t.text for t in soup.find('div', attrs={'class':'post-taglist'}).select('a')]
                titleStr = soup.find('div', attrs={'id':'question-header'}).select('a')[0].text
                textStr=''.join(map(str, soup.select('td.postcell div.post-text')[0].contents))
                title = title2Token(titleStr)
                body, code = body2Token(textStr)
                answers = ', '.join(trueTags)
                print answers
                textToken=[]
                textToken.append(list(set(title)))
                textToken.append(list(set(body)))
                textToken.append(list(set(code)))
                tags = tagWordAssociationMultiCV(textToken, nset=30, sgd='sgd')
            else:
                text = list(set(title2Token(raw)))
                tags = tagWordAssociationMultiCV([text], nset=8, sgd='', tit='body')
            return render_template('index.html', results=tags, rawdata=raw, answers=answers, errors=errors)
            #X_test = vectorizer.transform([raw.lower()])
            #tagProb = []
            #for keyn, cls in classifiers.items():
            #    dist_test = cls.decision_function(X_test)
            #    tagProb.append( 1.0/(1.0+np.exp(coef[keyn][0]*dist_test+coef[keyn][1])) )
            #idx=range(nkey)
            #idx.sort(lambda x,y: cmp(tagProb[x],tagProb[y]))
            #results = [allKeys[keyNum[idx[j]]] for j in range(3)]
    return render_template('index.html', errors=errors, results=results, rawdata=raw, answers=answers)


if __name__ == '__main__':
    #allKeys=[l.split()[0] for l in open('keywordsAll.txt').readlines()]
    #vectorizer = HashingVectorizer(decode_error='ignore', n_features=2 ** 20,
    #                               token_pattern=r"\b\w[\w#+.-]*(?<!\.$)",
    #                               binary=True, norm=None,
    #                               non_negative=True)
    #coef=loadCoef('LgCoef_body+title+code_partialfit_c.txt')
    #keyNum = [k for k,m in coef.items()]
    #nkey = len(keyNum)
    #classifiers = {}
    #for keyn in keyNum:
    #    classifiers[keyn] = joblib.load('data/SGD_key_'+str(keyn)+'_c.pkl')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
    s.close()
    app.run(host=ip)
