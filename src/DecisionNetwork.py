#Antonin Vobecky

import numpy as np
import glob
import os
import imageio as imio
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from math import ceil,floor
import shutil


def findThreshold(pos, neg):
    pos = [(x, 0) for x in pos]
    neg = [(x, 1) for x in neg]
    allAvg = pos + neg
    allAvg = sorted(allAvg, key=lambda tup: tup[1])

    bestIdx = 0
    bestVal = 0
    i = 0
    for thr in allAvg:
        correct = 0
        for avg in allAvg:
            if avg < thr and avg[1] == 0:
                correct += 1
            if avg > thr and avg[1] == 1:
                correct += 1
        if correct > bestVal:
            bestVal = correct
            bestIdx = i
        i += 1
    print('most correctly classified: ', bestVal, ' (', bestVal / len(allAvg), '%)')
    print('index of best threshold: ', bestIdx)
    print(allAvg[bestIdx])
    return allAvg[bestIdx][0]


def computeRes(set):
    from keras.models import model_from_json
    print('compute res')
    model_weights = "Network/model_1out_BN_DO-07_TF_1-improvement-05-0.68.hdf5"
    model_json = "Network/model_tensorflow_1out_BN_DO-07_TF_1.json"
    # load json and create model
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_weights)
    print("Loaded model from disk - val acc 0.68 after 5 epochs")

    # evaluate loaded model on test data
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    if set=='train':
        print('TRAINING SET')
        posBaseDir = '/storage/plzen1/home/vobecant/dataset_new/train/1/'
        negBaseDir = '/storage/plzen1/home/vobecant/dataset_new/train/0/'
    elif set=='val':
        print('VALIDATION SET')
        posBaseDir = '/storage/plzen1/home/vobecant/dataset_new/val/1/'
        negBaseDir = '/storage/plzen1/home/vobecant/dataset_new/val/0/'
    elif set == 'test':
        print('TEST SET')
        posBaseDir = 'Bags/'
        negBaseDir = 'Nothing/'

    if not os.path.exists('Nothing'):
        os.makedirs('Nothing')

    posFolders = next(os.walk(posBaseDir))[1]  # took help - not confident
    negFolders = next(os.walk(negBaseDir))[1]  # did not take help - confident

    i = 1
    posRes = []
    posImgs = 0
    posFoldersUsed = []

    for posFolder in posFolders:
        filelist = glob.glob(posBaseDir + str(posFolder) + '/*.png')
        filelist.sort()
        if len(filelist) < 30:
            continue
        posFoldersUsed.append(posFolder)
        print(i-1,posFolder)
        posImgs = posImgs + len(filelist)
        imgs = np.array([np.array(imio.imread(fname)) for fname in filelist])
        imgs = imgs[..., np.newaxis]
        res = model.predict_proba(imgs, batch_size=1, verbose=0)
        posRes.append(res[:, 0])
        print(i, "/", len(posFolders),' folder ',posFolder," with",
              len(filelist)," pictures")
        i = i + 1
    nPosCompleted = i
    print('completed ',i,' folders with pictures')
    print()

    i = 1
    negRes = []
    negImgs = 0
    for negFolder in negFolders:
        #if nPosCompleted == i:
        #    print('reached equal number of positive and negative bags')
        #    break
        filelist = glob.glob(negBaseDir + str(negFolder) + '/*.png')
        filelist.sort()
        if len(filelist) < 30:
            continue
        print(i-1+nPosCompleted, negFolder)
        negImgs = negImgs + len(filelist)
        imgs = np.array([np.array(imio.imread(fname)) for fname in filelist])
        imgs = imgs[..., np.newaxis]
        res = model.predict_proba(imgs, batch_size=1, verbose=0)
        negRes.append(res[:,0])
        print(str(i) + "/" + str(len(negFolders)) + '' + ' negative folder ' + str(negFolder) + " with " + str(
            len(filelist)) + " pictures")
        i = i + 1
    print('completed ', i, ' folders with pictures')
    print()

    res = {'pos': posRes, 'neg': negRes}
    

    bins = {'pos': posRes, 'neg': negRes}
    if set=='train':
        pickle.dump(bins, open('/storage/plzen1/home/vobecant/res_train126.p', 'wb'), protocol=2)
    elif set=='val':
        pickle.dump(bins, open('/storage/plzen1/home/vobecant/res_val126.p', 'wb'), protocol=2)
    elif set == 'test':
        pickle.dump(bins, open('./Network/res_test3.p', 'wb'), protocol=2)
        pickle.dump(posFoldersUsed, open('./Network/res_test3_folders.p', 'wb'), protocol=2)

    return res


def makeAllHists(pRes,nRes,nbins):
    posHist = np.array([])
    for p in pRes:
        posHist = np.append(posHist,makeHist(p,nbins))
    posHist = np.reshape(posHist,[-1,nbins]).tolist()
    y_pos = np.ones([len(posHist),])

    negHist = np.array([])
    for n in nRes:
        negHist = np.append(negHist,makeHist(n,nbins))
    negHist = np.reshape(negHist,[-1,nbins]).tolist()
    y_neg = np.zeros([len(negHist),])

    x = posHist+negHist
    y = np.append(np.ones([len(posHist), 1]), np.zeros([len(negHist), 1]))

    return x,y


def makeHist(res, nbins):
    bins = np.zeros([nbins,])
    for a in res:
        bins[findBin(a,nbins)]+=1
    bins = normalizeBin(bins)
    return bins


def findBin(value, nbins):
    step = 1.0/nbins
    cur_threshold = step
    for i in range(0,nbins):
        if value<cur_threshold:
            return i
        cur_threshold+=step


def normalizeBin(bins):
    nSamples = np.sum(bins)
    for i in range(0,len(bins)):
        bins[i] = bins[i] / nSamples
    return bins


def classif_kNN(x_train, y_train, x_test, y_test, n, ls):
    nbrs = KNeighborsClassifier(n_neighbors=n, algorithm='auto',leaf_size=ls).fit(x_train,y_train)
    classif = nbrs.predict(x_test)
    diffs = np.sum(abs(np.subtract(classif,y_test)))
    err = diffs/len(y_test)
    print('         n=',n,', error=',err)
    return err


def classif_svm(x_train, y_train, x_test, y_test,kernel,c,test):
    #svc = svm.LinearSVC(C=c)
    svc = svm.SVC(C=c,kernel=kernel,probability=True)
    svc.fit(x_train,y_train)
    classif = svc.predict(x_test)
    #probs = svc.predict_proba(x_test)
    dist = svc.decision_function(X=x_test)
    diffs = np.sum(abs(np.subtract(classif, y_test)))
    correct = np.where(np.subtract(classif, y_test)==0)
    correct = correct[0]
    distCorrect = dist[correct]
    distCorrect = np.abs(distCorrect)
    bestId = distCorrect.argsort()[-5:][::-1]
    bestDist = distCorrect[bestId]
    err = diffs / len(y_test)
    if test:
        print('         c=',c,kernel+' SVM error=', err)
    return err, bestId, bestDist

def genXY(bins):
    posBins = np.reshape(bins['pos'],[-1,10]).tolist()
    negBins = np.reshape(bins['neg'],[-1,10]).tolist()
    x = posBins+negBins
    y = np.append(np.ones([len(posBins),1]),np.zeros([len(negBins),1]))
    return x,y

def res2vec(res,poolType):
    '''
    Converts vecotor of results 'res' of variable length to vector of length K
    :param res: results
    :param k: length of output vector
    :return: vector of length k
    '''
    bins = [20,15,10]
    vec=[]
    for bin in bins:
        l = len(res)
        win = np.ceil(l/bin) # window
        win = int(win)
        str = np.floor(l/bin) # stride
        str = int(str)
        startId = 0
        endId = win
        if poolType=='max':
            while endId<l:
                vec.append(max(res[startId:endId]))
                startId+=str
                endId+=str
        elif poolType=='mean':
            while endId<l:
                vec.append(np.mean(res[startId:endId]))
                startId+=str
                endId+=str
        elif poolType=='median':
            while endId<l:
                vec.append(np.median(res[startId:endId]))
                startId+=str
                endId+=str
        else:
            print('Unsupported pooling type.')
    return vec


def spp(res, nbins, poolType):
    '''
    Converts vecotor of results 'res' of variable length to vector of length K
    :param res: results
    :param k: length of output vector
    :return: vector of length k
    '''
    vec=[]
    l = len(res)
    win = np.floor(l/nbins) + (l % nbins)# window
    win = int(win)
    str = np.floor(l/nbins) # stride
    str = int(str)
    startId = 0
    endId = win
    if poolType=='max':
        while endId<=l:
            vec.append(max(res[startId:endId]))
            startId+=str
            endId+=str
    elif poolType=='mean':
        while endId<=l:
            vec.append(np.mean(res[startId:endId]))
            startId+=str
            endId+=str
    elif poolType=='median':
        while endId<=l:
            vec.append(np.median(res[startId:endId]))
            startId+=str
            endId+=str
    else:
        print('Unsupported pooling type.')
    return vec






def test_spp(x_train, y_train, x_val, y_val, x_test, y_test):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    #kernels = ['linear']
    cs = [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5, 7, 10]
    best_kernel = ''
    best_c = 0
    lowest_err = 1
    bestId = []

    for kernel in kernels:
        for c in cs:
            err, bestId, bestDist = classif_svm(x_train=x_train, y_train=y_train, x_test=x_val, y_test=y_val, kernel=kernel, c=c,test=False)
            if err < lowest_err:
                lowest_err = err
                best_kernel=kernel
                best_c=c
    #print('lowest val error:',lowest_err)

    tst_err, bestId, bestDist = classif_svm(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, kernel=best_kernel, c=best_c, test=True)
    #print('test error of configuration with lowest val error:',tst_err)
    return tst_err, best_kernel, best_c, bestId, bestDist


def createConstOutput(res,outSize,poolType):
    pos_res = res['pos']
    neg_res = res['neg']
    x = []
    for r in pos_res:
        x.append(spp(r, nbins=outSize, poolType=poolType))
    for r in neg_res:
        x.append(spp(r, nbins=outSize, poolType=poolType))
    y = np.append(np.ones([len(pos_res), 1]), np.zeros([len(neg_res), 1]))
    return x,y


def showBestId(bestId):
    print(bestId)
    #posBaseDir = '/storage/plzen1/home/vobecant/dataset_new/TESTOVANI/1/'
    posBaseDir = 'Bags/'
    #negBaseDir = '/storage/plzen1/home/vobecant/dataset_new/TESTOVANI/0/'
    negBaseDir = 'Nothing/'

    posFolders = next(os.walk(posBaseDir))[1]  # took help - not confident
    negFolders = next(os.walk(negBaseDir))[1]  # did not take help - confident

    allFolders = np.array(posFolders+negFolders)
    allFoldersCorrect = []

    for posFolder in posFolders:
        if posFolder in bestId:
            filelist = glob.glob(posBaseDir + str(posFolder) + '/*.png')
            filelist.sort()
            if len(filelist) < 30:
                continue
            posFolder = posBaseDir + posFolder
            allFoldersCorrect.append(posFolder)
    '''
    nPos = len(allFoldersCorrect)

    for negFolder in negFolders:
        filelist = glob.glob(negBaseDir + str(negFolder) + '/*.png')
        if len(filelist) < 30:
            continue
        negFolder = negBaseDir + negFolder
        allFoldersCorrect.append(negFolder)

    allFoldersCorrect = np.array(allFoldersCorrect)
    bestFolders = allFoldersCorrect[bestId]
    '''
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    i=0
    for folder in allFoldersCorrect:

        filelist = glob.glob(str(folder) + '/*.png')
        tit = 'uncertain'
        filelist.sort()
        i+=1
        imgs = np.array([np.array(imio.imread(fname)) for fname in filelist])
        nrow = 7
        ncol = ceil((imgs.shape[0])/7)
        #print(nrow,ncol)
        fig = plt.figure(figsize=(ncol + 1, nrow + 1))
        plt.title(tit)
        gs = gridspec.GridSpec(nrow, ncol,
                               wspace=0.0, hspace=0.3,
                               top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                               left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
        for j in range(1,imgs.shape[0]):
            im = imgs[j, :, :]
            #print(floor((i+1)/ncol),floor((i+1)/nrow))
            ax = plt.subplot(nrow,ncol,j)
            ax.imshow(im, cmap='gray')
            plt.axis('off')
            #plt.title(res[i, 0])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        #fig.title(tit)
        plt.show()


# Find sections with decision ambiguity
def TestWholeNewFolder(sequenceObtained, parametersFound, showPictures):
    if sequenceObtained == False:
        computeRes('test')
    best_err = 1
    best_size=0
    best_kernel = ''
    best_c = 0
    best_pool = ''
    bestId = []
    bestDist = []
    outSizeMax = 30
    poolTypes = ['mean','max']
    train_res = np.load('Network/res_train.p',allow_pickle=True)
    test_res = np.load('Network/res_test3.p',allow_pickle=True)
    folders = np.load('./Network/res_test3_folders.p',allow_pickle=True)

    best_pool = 'max'
    best_size = 13
    if parametersFound == False:
        for size in range(1,outSizeMax+1):
            for pool in poolTypes:
                print('size:',size,'pooling type:',pool)
                x_train, y_train = createConstOutput(train_res,size,pool)
                x_val, y_val = createConstOutput(val_res,size,pool)
                x_test, y_test = createConstOutput(test_res,size,pool)
                tst_err, tst_kernel, tst_c, id, dist = test_spp(x_train,y_train,x_val,y_val,x_test,y_test)
                if tst_err<best_err:
                    bestId = id
                    best_err=tst_err
                    best_size = size
                    best_kernel=tst_kernel
                    best_c=tst_c
                    best_pool = pool
                    best_dist = dist
        print('best error:',best_err,'best size:',best_size,'best pool:',best_pool,'best kernel:',best_kernel,'best c:',best_c)
        print(bestId) # [18 29  7 27 22] for best error: 0.244444444444 best size: 13 best pool: max best kernel: poly best c: 0.3
    


    x_train, y_train = createConstOutput(train_res,poolType=best_pool, outSize=best_size)
    x_test, y_test = createConstOutput(test_res, poolType=best_pool, outSize=best_size)
    svc = svm.SVC(C=0.3, kernel='poly', probability=True)
    svc.fit(x_train, y_train)
    classif = svc.predict(x_test)
    dist = svc.decision_function(X=x_test)

    print(classif)
    print(folders)
    bestId = []
    noDecisionAmbiguity = []
    cnt = 0
    for clas in classif:
        if classif[cnt] >= 1:
            bestId.append(folders[cnt])
        else:
            noDecisionAmbiguity.append(folders[cnt])
        cnt += 1

    #Save classified facial images for output video

    array = []
    with open('Network/locations.p', 'rb') as f:
        array = pickle.load(f)
    rectangles = []
    rectanglesNoAmbiguity = []
    for folder in bestId:
        images = os.listdir('Bags/'+folder)
        for image in images:
            tmpNumber = image.split('.')
            tmpNumber = int(tmpNumber[0])
            for location in array[tmpNumber]:
                if location != [] and str(location[0]) == str(folder):
                    rectangles.append([tmpNumber,location[2],location[3],location[4],location[5],str(folder)])

    for folder in noDecisionAmbiguity:
        images = os.listdir('Bags/'+folder)
        for image in images:
            tmpNumber = image.split('.')
            tmpNumber = int(tmpNumber[0])
            for location in array[tmpNumber]:
                if location != [] and str(location[0]) == str(folder):
                    rectanglesNoAmbiguity.append([tmpNumber,location[2],location[3],location[4],location[5],str(folder)])

    if showPictures == True:
        showBestId(bestId)

    print('Done locating sections with decision ambiguity')
    with open('Network/rectangles.p', 'wb') as f:
        pickle.dump(rectangles,f)

    with open('Network/rectanglesNoDecisionAmbiguity.p', 'wb') as f:
        pickle.dump(rectanglesNoAmbiguity,f)

    if os.path.exists('Nothing'):
        shutil.rmtree('Nothing', ignore_errors=False, onerror=None)

    if os.path.exists('Bags'):
        shutil.rmtree('Bags', ignore_errors=False, onerror=None)

    return rectangles, rectanglesNoAmbiguity

def ShowOnlySurvey(showPictures):
    from src.ShowHistory import load_data_survey

    best_pool = 'max'
    best_size = 13
    train_res = np.load('res_train.p',allow_pickle=True)
    test_res = np.load('res_test20.p',allow_pickle=True)
    x_train, y_train = createConstOutput(train_res, poolType=best_pool, outSize=best_size)
    x_test, y_test = createConstOutput(test_res, poolType=best_pool, outSize=best_size)
    test = {'x': x_test, 'y': y_test}
    import pickle
    with open('Network/test_survey.pickle', 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('survey saved')
    svc = svm.SVC(C=0.3, kernel='poly', probability=True)
    svc.fit(x_train, y_train)
    classif = svc.predict(x_test)
    # probs = svc.predict_proba(x_test)
    dist = svc.decision_function(X=x_test)
    diffs = np.sum(abs(np.subtract(classif, y_test)))
    correct = np.where(np.subtract(classif, y_test) == 0)
    correct = correct[0]
    distCorrect = dist[correct]
    distCorrect = np.abs(distCorrect)
    bestId = distCorrect.argsort()[-5:][::-1]
    bestDist = distCorrect[bestId]
    err = diffs / len(y_test)

    print('bestIDs:',bestId)
    print('err:', err)
    if showPictures == True:
        bestId = np.array([13, 11, 14, 10,  8])
        showBestId(bestId)
