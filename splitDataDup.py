import csv
import json, gzip, os
import random

def splitCsv(fileName, nPart):
    with open(fileName, 'rb') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        plist = []
        pdic = {}
        data.next() # skip first line
        for row in data:
            pt = row[1]+' '+str(len(row[2]))+' '+row[2].split()[-1]+' '+row[3]
            plist.append(pt)
            pdic[pt] = 0
        uplist=list(set(plist))
        n = len(uplist)
        i = 1
        n1 = 0
        n2 = n/nPart + 1
        print n1, n2, n, len(plist)
        fileOpen = False
        csvfile.seek(0)
        data.next() # skip first line
        for k, row in enumerate(data):
            if not fileOpen:
                outputName = fileName+'.rdup.'+str(i)
                outfile = open(outputName, "wb")
                writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
                fileOpen = True
            if pdic[plist[k]] == 1:
                continue
            else:
                pdic[plist[k]] = 1
            writer.writerow(row)
            n1 += 1
            if n1 == n2:
                print outputName
                i += 1
                outfile.close()
                fileOpen = False
                n1 = 0
        if fileOpen:
            print outputName
            outfile.close()

def makeTestSet():
    #split to 1/10, 1/10, 2/10, 2/10, 4/10
    seed = 123
    data = json.load(gzip.open('Train.rdup.10.json.gz'))
    feed1=[]
    feed2=[]
    feed3=[]
    feed4=[]
    feed5=[]
    random.seed(seed)
    for d in data:
        p=random.random()
        if p<0.1:
            feed1.append({'id' : d['id'], 'title': d['title'], 'body' : d['body'], 'code': d['code'], 'tag' : d['tag']})
        elif p<0.2:
            feed2.append({'id' : d['id'], 'title': d['title'], 'body' : d['body'], 'code': d['code'], 'tag' : d['tag']})
        elif p<0.4:
            feed3.append({'id' : d['id'], 'title': d['title'], 'body' : d['body'], 'code': d['code'], 'tag' : d['tag']})
        elif p<0.6:
            feed4.append({'id' : d['id'], 'title': d['title'], 'body' : d['body'], 'code': d['code'], 'tag' : d['tag']})
        else:
            feed5.append({'id' : d['id'], 'title': d['title'], 'body' : d['body'], 'code': d['code'], 'tag' : d['tag']})

    with open('test1.json', 'w') as outfile:
        json.dump(feed1, outfile)
    with open('test2.json', 'w') as outfile:
        json.dump(feed2, outfile)
    with open('test3.json', 'w') as outfile:
        json.dump(feed3, outfile)
    with open('test4.json', 'w') as outfile:
        json.dump(feed4, outfile)
    with open('test5.json', 'w') as outfile:
        json.dump(feed5, outfile)
    os.system('gzip test1.json')
    os.system('gzip test2.json')
    os.system('gzip test3.json')
    os.system('gzip test4.json')
    os.system('gzip test5.json')





if __name__ == '__main__':
    #splitCsv('Train.csv', 10)
    makeTestSet()
