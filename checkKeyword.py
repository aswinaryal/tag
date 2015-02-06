import csv, gzip
import sys
#import numpy as np

def getAllKeys(n):
    keywords = []
    filename = 'Train.csv.rdup.'+str(n)+'.gz'
    data = csv.reader(gzip.open(filename), delimiter=',')
    for row in data:
        for key in row[3].split():
            keywords.append(key)
    ukey=list(set(keywords))
    ukeynum=[keywords.count(k) for k in ukey]
    idx=range(len(ukey))
    idx.sort(lambda x,y: cmp(ukeynum[x],ukeynum[y]))
    outfile=open('keywords'+str(n)+'.txt','w')
    for i in idx[::-1]:
        outfile.write("%s %d\n" % (ukey[i],ukeynum[i]))
    outfile.close()

    #ukeylen=np.array(ukeylen)
    #idx=np.nonzero(ukeylen==1)
    #for i in idx[0]:
    #    print i, ukey[i]


getAllKeys(int(sys.argv[1]))
