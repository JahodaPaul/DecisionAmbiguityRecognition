#Antonin Vobecky

# import matplotlib.pyplot as plt
import imageio
import numpy as np
# import matplotlib.pyplot as plt
import json
import pickle


def showHistory(fname):
    # with open(fname) as json_data:
    #    history = json.load(json_data)
    #    print(history)

    history = pickle.load(open(fname, "rb"))

    # list all data in history
    plt.figure()
    print(history.keys())
    # summarize history for accuracy
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def testNetworkImages(model_weights, model_json, img_path):
    import keras.optimizers as optimizers
    from keras.models import model_from_json
    # model = load_model(model_h5)
    # load json and create model
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_weights)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # img = imageio.imread(img_path)
    # img = np.resize(img,[1,100,100,1])
    img = np.load('tst_neg_2.p')
    # img = np.load('63_test_negative_np.p')
    # img = np.transpose(img,(2,0,1))
    img = img[..., np.newaxis]
    print('image loaded')
    res = model.predict_proba(img, batch_size=1, verbose=1)
    # print(res)
    min = (res[:, 0]).argsort()
    max = (-res[:, 0]).argsort()

    from matplotlib import gridspec
    nrow = 11
    ncol = 14
    fig = plt.figure(figsize=(ncol + 1, nrow + 1))
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.3,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))
    for i in range(nrow):
        for j in range(ncol):
            im = img[i * nrow + j, :, :, 0]
            ax = plt.subplot(gs[i, j])
            ax.imshow(im, cmap='gray')
            plt.axis('off')
            plt.title(res[i, 0])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.show()

    skip = 0
    plt.figure()
    for i in range(0, 20):
        plt.subplot(4, 5, i + 1)
        plt.axis('off')
        plt.title(str(res[min[i + i * skip], 0]))
        plt.imshow(img[min[i + i * skip], :, :, 0], cmap='gray')

    plt.figure()
    for i in range(0, 20):
        plt.subplot(4, 5, i + 1)
        plt.axis('off')
        plt.title(str(res[max[i + i * skip], 0]))
        plt.imshow(img[max[i + i * skip], :, :, 0], cmap='gray')

    import pandas as pd
    df = pd.DataFrame(res[:, 0])
    xAx = np.arange(0, img.shape[0], 1)
    plt.figure()
    plt.plot(xAx, res[:, 0])
    plt.title('original - unfiltered')
    plt.figure()
    mAvg = df.rolling(window=5).mean()
    plt.plot(xAx, mAvg)
    plt.title('window=5')
    plt.figure()
    mAvg = df.rolling(window=10).mean()
    plt.plot(xAx, mAvg)
    plt.title('window=10')
    plt.figure()
    mAvg = df.rolling(window=15).mean()
    plt.plot(xAx, mAvg)
    plt.title('window=15')


def testNetworkSet(model_json, model_weights):
    from keras.models import model_from_json
    from keras import optimizers
    from SpatialPyramidPooling import SpatialPyramidPooling
    from keras.layers.wrappers import TimeDistributed
    from keras.layers import Dropout
    balanced = True

    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})

    from net_bags3_new import load_data
    # x_test, y_test = load_data('test', balanced)
    x_test20, y_test20 = load_data_survey()
    y_test20_2 = [y[0] for y in y_test20]
    assert len(x_test20) == len(y_test20), "length of test X and Y is not equal"

    from MyGenerator import DataGenerator
    gen = DataGenerator(False)
    test_generator = gen.generate(x_in=x_test20, y_in=y_test20)

    x_val, y_val = load_data('val', True)
    y_val2 = [y[0][1] for y in y_val]
    val_generator = gen.generate(x_in=x_val, y_in=y_val)

    x_test, y_test = load_data('test', True)
    y_test2 = [y[0] for y in y_test]
    test_generator = gen.generate(x_in=x_test, y_in=y_test)

    print('start testing...')
    for weights in model_weights:
        model.load_weights(weights)
        for layer in model.layers:
            # print(type(layer))
            if type(layer) in [TimeDistributed]:
                # print(type(layer.layer))
                if type(layer.layer) in [Dropout]:
                    layer.layer.rate = 0.0
                    # print(layer.layer,layer.layer.rate)
        learningRate = 0.0000001
        optimizer = optimizers.adam(lr=learningRate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print("")
        print("test 20")
        for i in range(0, 3):
            # load weights into new model
            score1 = model.evaluate_generator(test_generator, steps=len(x_test20),workers=1,use_multiprocessing=False)
            acc = 0
            for i in range(0, 20):
                res = model.test_on_batch(x_test20[i],y_test20[i])#[0][0]
                acc+=res[1]
            acc = acc/20
            print('acc',acc)
            print('file:', weights, ";test: %s: %.2f%%" % (model.metrics_names[1], score1[1] * 100), 'acc:', acc)
        print("val")
        for i in range(0, 1):
            # load weights into new model
            score1 = model.evaluate_generator(val_generator, steps=len(x_val))
            acc = 0
            for i in range(0, len(y_val)):
                res = model.test_on_batch(x_val[i], y_val[i])  # [0][0]
                acc += res[1]
            acc =  acc / len(y_val)
            print('acc',acc)
            print('file:', weights, ";test: %s: %.2f%%" % (model.metrics_names[1], score1[1] * 100), 'acc:', acc)

        print("test")
        for i in range(0, 1):
            # load weights into new model
            score1 = model.evaluate_generator(test_generator, steps=len(x_test))
            acc = 0
            for i in range(0, len(y_test)):
                res = model.test_on_batch(x_test[i], y_test[i])  # [0][0]
                acc += res[1]
            acc = acc / len(y_test)
            print('acc', acc)
            print('file:', weights, ";test: %s: %.2f%%" % (model.metrics_names[1], score1[1] * 100), 'acc:', acc)


def load_data_survey():
    posBaseDir = '/storage/plzen1/home/vobecant/dataset_new/TESTOVANI/1/'
    #posBaseDir = 'C:/Users/Antonín/škola/Magisterské studium/CMP/face_recognition/CNN/FOLDER/TESTOVANI/1/'
    negBaseDir = '/storage/plzen1/home/vobecant/dataset_new/TESTOVANI/0/'
    #negBaseDir = 'C:/Users/Antonín/škola/Magisterské studium/CMP/face_recognition/CNN/FOLDER/TESTOVANI/0/'

    pos_filelist = [posBaseDir + "23/23_pos.p", posBaseDir + "138/138_pos.p", posBaseDir + "275/275_pos.p",
                    posBaseDir + "276/276_pos.p", posBaseDir + "303/303_pos.p", posBaseDir + "344/344_pos.p",
                    posBaseDir + "391/391_pos.p", posBaseDir + "392/392_pos.p", posBaseDir + "434/434_pos.p",
                    posBaseDir + "10032/10032_pos.p"]
    neg_filelist = [negBaseDir + "57/57_pos.p", negBaseDir + "58/58_pos.p", negBaseDir + "59/59_pos.p",
                    negBaseDir + "60/60_pos.p", negBaseDir + "63/63_pos.p", negBaseDir + "382/382_pos.p",
                    negBaseDir + "691/691_pos.p", negBaseDir + "698/698_pos.p", negBaseDir + "855/855_pos.p",
                    negBaseDir + "856/856_pos.p"]

    from keras.utils.np_utils import to_categorical
    yp = to_categorical(1, 2)
    yn = to_categorical(0, 2)

    nOutputs = 2
    x_pos = []
    y_pos = []
    for p in pos_filelist:
        xp = np.load(p)
        if xp.shape[0] < 30 or xp.shape[0] > 300:
            continue
        xp = np.transpose(xp[np.newaxis, ...], (1, 0, 2, 3, 4))
        x_pos.append(xp)
        y_pos.append(np.resize(yp, (xp.shape[0], nOutputs)))
    l = np.array([len(x) for x in x_pos])
    sortIdx = np.flip(l.argsort(), axis=-1)
    x_pos = [x_pos[i] for i in sortIdx]
    y_pos = [y_pos[i] for i in sortIdx]

    x_neg = []
    y_neg = []
    for n in neg_filelist:
        xn = np.load(n)
        if xn.shape[0] < 30 or xn.shape[0] > 300:
            continue
        xn = np.transpose(xn[np.newaxis, ...], (1, 0, 2, 3, 4))
        x_neg.append(xn)
        y_neg.append(np.resize(yn, (xn.shape[0], nOutputs)))
    l = np.array([len(x) for x in x_neg])
    sortIdx = np.flip(l.argsort(), axis=-1)
    x_neg = [x_neg[i] for i in sortIdx]
    y_neg = [y_neg[i] for i in sortIdx]

    n_p = len(x_pos)
    n_n = len(x_neg)
    y = []
    if n_p > n_n:  # more positive bags
        print(set, 'more positive bags:', n_p, 'vs', n_n)
        x = x_pos[:n_n] + x_neg
        y = y_pos[:n_n] + y_neg
    elif n_n > n_p:  # more negative bags
        print(set, 'more negative bags:', n_n, 'vs', n_p)
        x = x_pos + x_neg[:n_p]
        y = y_pos + y_neg[:n_p]
    else:  # equal number of positive and negative bags
        print(set, 'equal number of positive and negative bags:', n_p, 'and', n_n)
        x = x_pos + x_neg
        y = y_pos + y_neg

    print('len y:', len(y), 'shape y0:', y[0].shape)

    print('number of bags:', len(x), 'length of labels:', len(y))

    for i in range(0, len(x)):
        if not (x[i].shape[0] == y[i].shape[0]):
            print('In x in', i, 'are not equal.')

    return x, y


if __name__ == '__main__':
    base = "/storage/plzen1/home/vobecant/learned_networks/out1_BN_DO-02_TF_pool15_2FC/"
    model_weights = [base + "model_1out_BN_DO-07_TF_1-improvement-884-0.68.hdf5",
                     base + "model_1out_BN_DO-07_TF_1-improvement-897-0.71.hdf5",
                     base + "model_1out_BN_DO-07_TF_1-improvement-898-0.67.hdf5",
                     base + "model_1out_BN_DO-07_TF_1-improvement-925-0.74.hdf5",
                     base + "model_1out_BN_DO-07_TF_1-improvement-927-0.70.hdf5",
                     base + "model_1out_BN_DO-07_TF_1-improvement-937-0.64.hdf5",
                     base + "model_1out_BN_DO-07_TF_1-improvement-942-0.71.hdf5",
                     base + "model_1out_BN_DO-07_TF_1-improvement-974-0.67.hdf5",
                     base + "model_1out_BN_DO-07_TF_1-improvement-996-0.67.hdf5"]
    model_json = base + "model_tensorflow_1out_BN_DO-07_TF_1.json"
    img_path = "/storage/plzen1/home/vobecant/test_neg.png"
    # testNetworkImages(model_weights,model_json=model_json, img_path=img_path)
    # testNetworkSet(model_json,model_name)
    # showHistory('history_tensorflow_1out_BN_DO-07_TF_1_part3')
    # showHistory('history_bags_tanh_part1')
    testNetworkSet(model_json=model_json, model_weights=model_weights)
