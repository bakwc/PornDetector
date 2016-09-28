#!/usr/bin/env python

import os
import sys
import zlib
import cv2
import time
import random
from threading import Thread
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
try:
    import cPickle as pickle
    from urllib2 import urlopen
    from Queue import Queue
except ImportError:
    import pickle
    from urllib.request import urlopen
    from queue import Queue

FILE_LOAD_THREADS = 1
FILE_SEED = 24
CLUSTER_SEED = 24
CLUSTERS_NUMBER = 1000
BAYES_ALPHA = 0.1
ADA_BOOST_ESTIMATORS = 110
VERBOSE = True
USE_CACHE = True

_g_removed = False


class PCR:

    def __init__(self):
        self.__clustersNumber = CLUSTERS_NUMBER
        self.__queue = Queue()
        self.__verbose = VERBOSE
        self.__useCache = USE_CACHE

        for i in range(FILE_LOAD_THREADS):
            t = Thread(target=self.__worker)
            t.daemon = True
            t.start()

        self.__kmeans = MiniBatchKMeans(
            n_clusters=self.__clustersNumber,
            random_state=CLUSTER_SEED,
            verbose=self.__verbose)
        self.__tfidf = TfidfTransformer()
        self.__tfidf1 = TfidfTransformer()

        self.__clf = AdaBoostClassifier(MultinomialNB(alpha=BAYES_ALPHA), n_estimators=ADA_BOOST_ESTIMATORS)
        self.__clf1 = AdaBoostClassifier(MultinomialNB(alpha=BAYES_ALPHA), n_estimators=ADA_BOOST_ESTIMATORS)

    def __worker(self):
        while True:
            task = self.__queue.get()
            func, args = task
            try:
                func(args)
            except Exception as e:
                print('EXCEPTION:', e)
            self.__queue.task_done()

    def train(self, positiveFiles, negativeFiles):
        cachedData = self.__loadCache()
        if cachedData is None:
            self.__log('loading positives')
            positiveSamples = self.__loadSamples(positiveFiles)
            self.__log('loading negatives')
            negativeSamples = self.__loadSamples(negativeFiles)

            totalDescriptors = []
            self.__addDescriptors(totalDescriptors, positiveSamples)
            self.__addDescriptors(totalDescriptors, negativeSamples)

            self.__kmeans.fit(totalDescriptors)
            clusters = self.__kmeans.predict(totalDescriptors)

            self.__printDistribution(clusters)
            self.__saveCache((positiveSamples, negativeSamples, self.__kmeans, clusters))
        else:
            self.__log('using cache')
            positiveSamples, negativeSamples, self.__kmeans, clusters = cachedData

        totalSamplesNumber = len(negativeSamples) + len(positiveSamples)
        counts = lil_matrix((totalSamplesNumber, self.__clustersNumber))
        counts1 = lil_matrix((totalSamplesNumber, 256))
        self.__currentSample = 0
        self.__currentDescr = 0
        self.__calculteCounts(positiveSamples, counts, counts1, clusters)
        self.__calculteCounts(negativeSamples, counts, counts1, clusters)
        counts = csr_matrix(counts)
        counts1 = csr_matrix(counts1)

        self.__log('training bayes classifier')
        tfidf = self.__tfidf.fit_transform(counts)
        tfidf1 = self.__tfidf1.fit_transform(counts1)
        classes = [True] * len(positiveSamples) + [False] * len(negativeSamples)
        self.__clf.fit(tfidf, classes)
        self.__clf1.fit(tfidf1, classes)

        self.__log('training complete')

    def predict(self, files):
        self.__log('loading files')
        samples = self.__loadSamples(files)
        totalDescriptors = []
        self.__addDescriptors(totalDescriptors, samples)
        self.__log('predicting classes')
        clusters = self.__kmeans.predict(totalDescriptors)
        counts = lil_matrix((len(samples), self.__clustersNumber))
        counts1 = lil_matrix((len(samples), 256))
        self.__currentSample = 0
        self.__currentDescr = 0
        self.__calculteCounts(samples, counts, counts1, clusters)
        counts = csr_matrix(counts)
        counts1 = csr_matrix(counts1)

        tfidf = self.__tfidf.transform(counts)
        tfidf1 = self.__tfidf1.transform(counts1)

        self.__log('classifying')

        weights = self.__clf.predict_log_proba(tfidf.toarray())
        weights1 = self.__clf1.predict_log_proba(tfidf1.toarray())
        predictions = []
        for i in range(0, len(weights)):
            w = weights[i][0] - weights[i][1]
            w1 = weights1[i][0] - weights1[i][1]

            pred = w < 0
            pred1 = w1 < 0

            if pred != pred1:
                pred = w + w1 < 0

            predictions.append(pred)

        self.__log('prediction complete')
        return predictions

    def saveModel(self, fileName):
        data = pickle.dumps((self.__clustersNumber, self.__kmeans, self.__tfidf,
                             self.__tfidf1, self.__clf, self.__clf1), -1)
        data = zlib.compress(data)
        open(fileName, 'wb').write(data)

    def loadModel(self, fileName):
        data = open(fileName, 'rb').read()
        data = zlib.decompress(data)
        data = pickle.loads(data)
        self.__clustersNumber, self.__kmeans, self.__tfidf, self.__tfidf1, self.__clf, self.__clf1 = data

    def __log(self, message):
        if self.__verbose:
            print(message)

    def __saveCache(self, data):
        if not self.__useCache:
            return
        data = pickle.dumps(data, -1)
        data = zlib.compress(data)
        open('cache.bin', 'w').write(data)

    def __loadCache(self):
        if not self.__useCache:
            return None
        if not os.path.isfile('cache.bin'):
            return None
        data = open('cache.bin', 'r').read()
        data = zlib.decompress(data)
        data = pickle.loads(data)
        return data

    def __calculteCounts(self, samples, counts, counts1, clusters):
        cn = self.__clustersNumber
        for s in samples:
            currentCounts = {}
            for d in s[0]:
                currentCounts[clusters[self.__currentDescr]] = currentCounts.get(clusters[self.__currentDescr], 0) + 1
                self.__currentDescr += 1
            for clu, cnt in currentCounts.iteritems():
                counts[self.__currentSample, clu] = cnt
            for i, histCnt in enumerate(s[1]):
                counts1[self.__currentSample, i] = histCnt[0]
            self.__currentSample += 1

    def __printDistribution(self, clusters):
        if not self.__verbose:
            return
        distr = {}
        for c in clusters:
            distr[c] = distr.get(c, 0) + 1
        v = sorted(distr.values(), reverse=True)
        print('distribution:', v[0:15], '...', v[-15:])

    def __addDescriptors(self, totalDescriptors, samples):
        for sample in samples:
            for descriptor in sample[0]:
                totalDescriptors.append(descriptor)

    def __loadSamples(self, files):
        samples = [[]] * len(files)
        n = 0
        for f in files:
            self.__queue.put((self.__loadSingleSample, (f, samples, n)))
            n += 1
        self.__queue.join()
        if _g_removed:
            print(' === REMOVED = TERMINATE')
            sys.exit(44)
        return samples

    def __loadSingleSample(self, args):
        global _g_removed
        fileName, samples, sampleNum = args
        des, hist = self.__getFeatures(fileName)
        if des is None:
            print('ERROR: failed to load', fileName)
            os.remove(fileName)
            _g_removed = True
            # sys.exit(44)
            des = []
            hist = [[0]] * 256
        samples[sampleNum] = (des, hist)

    def __getFeatures(self, fileName):
        fid = 'cache/' + str(zlib.crc32(fileName))
        self.__log('loading %s' % fileName)
        if os.path.isfile(fid):
            des, hist = pickle.loads(open(fid, 'rb').read())
        else:
            img = cv2.imread(fileName)

            if img.shape[1] > 1000:
                cf = 1000.0 / img.shape[1]
                newSize = (int(cf * img.shape[0]), int(cf * img.shape[1]), img.shape[2])
                img.resize(newSize)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            s = cv2.SIFT(nfeatures=400)

            d = cv2.DescriptorExtractor_create("OpponentSIFT")
            kp = s.detect(gray, None)
            kp, des = d.compute(img, kp)

            hist = self.__getColorHist(img)

            #open(fid, 'wb').write(pickle.dumps((des, hist), -1))

        return des, hist

    def __getColorHist(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
        return dist


def loadDir(dirName):
    files = os.listdir(dirName)
    fnames = []
    for f in files:
        if not f.endswith('.jpg'):
            continue
        fileName = dirName + '/' + f
        fnames.append(fileName)
    return fnames


def loadFileLists():
    random.seed(FILE_SEED)

    positiveFiles = sorted(loadDir('2'))
    negativeFiles = sorted(loadDir('1'))

    random.shuffle(positiveFiles)
    random.shuffle(negativeFiles)

    minLen = min(len(positiveFiles), len(negativeFiles))

    p20 = int(0.2 * minLen)

    testFiles = positiveFiles[:p20] + negativeFiles[:p20]
    positiveFiles = positiveFiles[p20:]
    negativeFiles = negativeFiles[p20:]

    print(testFiles[0], negativeFiles[0], positiveFiles[0])

    testFiles = loadDir('1test')

    return positiveFiles, negativeFiles, testFiles


def train():
    positiveFiles, negativeFiles, testFiles = loadFileLists()

    pcr = PCR()
    pcr.train(positiveFiles, negativeFiles)
    pcr.saveModel('model.bin')


def predict():

    positiveFiles, negativeFiles, testFiles = loadFileLists()
    testFiles = testFiles

    pcr = PCR()
    pcr.loadModel('model.bin')
    pred = pcr.predict(testFiles)
    total = 0
    correct = 0

    for i in xrange(0, len(testFiles)):
        isCorrect = ((testFiles[i][0] == '1' and not pred[i]) or (testFiles[i][0] == '2' and pred[i]))

        print(isCorrect, pred[i], testFiles[i])
        # if not isCorrect:
        #  print testFiles[i]
        correct += int(isCorrect)

        total += 1
    print('sum: \t', float(correct) / total)


def predictTest():
    files = ['test.jpg']
    pcr = PCR()
    pcr.loadModel('model.bin')
    pred = pcr.predict(files)
    print('\n\n ===', pred[0], '===\n\n')


def predictUrl(url):
    f = open('test.jpg', 'wb')
    f.write(urlopen(url).read())
    f.close()
    time.sleep(0.5)
    predictTest()


def printUsage():
    print('Usage: ')
    print('  %s train                              - train model' % sys.argv[0])
    print('  %s url http://sample.com/img.jpg      - check given url' % sys.argv[0])
    sys.exit(42)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        printUsage()
    mode = sys.argv[1]
    if mode == 'train':
        train()
        time.sleep(0.5)
        predict()
    elif mode == 'url':
        if len(sys.argv) < 3:
            printUsage()
        url = sys.argv[2]
        predictUrl(url)
    else:
        printUsage()
