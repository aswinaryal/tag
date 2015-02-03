import csv, gzip
import lxml.html
import json
import re
import sys
#from nltk.stem import PorterStemmer

def body2Token(text):
    stopWords={s.strip() for s in open('data/stopwordsMod.txt').readlines()}
    bodyToken = []
    codeToken = []
    try:
        body = lxml.html.fromstring(text.decode('utf-8', 'ignore'))
    except ValueError:
        return bodyToken, codeToken
    code = ''
    for element in body.iterdescendants():
        if element.tag == 'code':
            code += ' ' + element.text_content() # makes sure there is a space so regex can split
            element.drop_tree()
    mainBody = body.text_content()
    for w in re.findall(TEXT_RE, mainBody.lower()):
        if w != '':
            if w not in stopWords:
                #bodyToken.append(stemmer.stem(w))
                if len(w) > 2 and w[-2:] == "'s":
                    w=w[:-2]
                bodyToken.append(w)
    for w in re.findall(CODE_RE, code.lower()):
        if w != '':
            codeToken.append(w)
        codeToken=list(set(codeToken))
    return bodyToken, codeToken

def title2Token(text):
    stopWords={s.strip() for s in open('data/stopwordsMod.txt').readlines()}
    tToken = []
    for w in re.findall(TEXT_RE, text.lower()):
        if w != '':
            if w not in stopWords:
                if len(w) > 2 and w[-2:] == "'s":
                    w=w[:-2]
                tToken.append(w)
    return tToken

def cleanData(n):
    data = csv.reader(gzip.open('Train.csv.rdup.'+str(n)+'.gz'), delimiter=',')
    if n==1:
        data.next()
    feed=[]
    for row in data:
        title = title2Token(row[1])
        body, code = body2Token(row[2])
        tag = row[3].split()
        feed.append({'id' : int(row[0]), 'title': title, 'body' : body, 'code': code, 'tag' : tag})
    with open('Train.rdup.'+str(n)+'.json', 'w') as outfile:
        json.dump(feed, outfile)


#stemmer = PorterStemmer()
CODE_RE = re.compile(r"[a-z#]{1}[-\w'+]+[a-z+]")
TEXT_RE = re.compile(r"([A-Za-z#][-\w'+]*[A-Za-z+]|[A-Za-z]+#|[A-Za-z]{1})")

if __name__ == '__main__':
    cleanData(int(sys.argv[1]))
