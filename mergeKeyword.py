from collections import defaultdict
import numpy as np
import json, gzip

def mergeKeys():
    keyDict=defaultdict(int)
    for i in range(1,11):
        for line in open('keywords'+str(i)+'.txt').readlines():
            lstr = line.split()
            keyDict[lstr[0]] += int(lstr[1])

    outfile=open('keywordsAll.txt','w')
    for k in sorted(keyDict, key=keyDict.get, reverse=True):
        outfile.write("%s %d\n" % (k,keyDict[k]))
    outfile.close()

def keywordFraction():
    allKeys=[l.split()[0] for l in open('keywordsAll.txt').readlines()]
    keyOcc=[int(l.split()[1]) for l in open('keywordsAll.txt').readlines()]

    for x in [500, 1000, 1500, 2000, 2500]:
        coverage=0
        keys = allKeys[0:x]
        data = json.load(gzip.open('Train.rdup.3.json.gz'))
        for d in data:
            for t in d['tag']:
                if t in keys:
                    coverage+=1
                    break
        print x, coverage, 1.0*coverage/len(data)



    for x in [1000, 100, 10, 1]:
        for i in xrange(len(allKeys)):
            if keyOcc[i] == x:
                print "top ", i , " tags occur >", x
                break

#mergeKeys()
keywordFraction()
