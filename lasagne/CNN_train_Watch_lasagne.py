import theano
import theano.tensor as T
import lasagne
import numpy as np
import time
import os
import cPickle as pickle
import gzip
from collections import OrderedDict

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start : minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def seq_padding(seq, maxlen):
    # this function pads seq in each dimension along 2 directions(head and tail)
    # please make sure length of seq is not larger than maxlen
    if seq.shape[0] > maxlen:
        print seq.shape[0], maxlen
        raise Exception("seq_padding: seq is larger than maxlen")
    else:
        head_len = (maxlen-seq.shape[0])//2
        tail_len = maxlen-seq.shape[0]-head_len
        head = np.array([seq[0] for _ in range(head_len)])
        tail = np.array([seq[-1] for _ in range(tail_len)])
        res = seq
        if head.shape[0] != 0:
            res = np.concatenate([head, res])
        if tail.shape[0] != 0:
            res = np.concatenate([res, tail])
        assert res.shape[0] == maxlen
        return res

def load_data_N_folds(path, default_path, test_id = 0, valid_portion=0.1, sort_by_len=True):
    def get_dataset_file(dataset, default_dataset):
        data_dir, data_file = os.path.split(dataset)
        if data_dir == "" and not os.path.isfile(dataset):
            # Check if dataset is in the data directory.
            new_path = os.path.join(os.path.split(__file__)[0],"data",dataset)
            if os.path.isfile(new_path) or data_file == default_dataset:
                dataset = new_path
        return dataset

    def Has(bigger_list, smaller_list):
        for string in smaller_list:
            if string in bigger_list:
                return True
        return False

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    path = get_dataset_file(path, default_path)
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    folds = pickle.load(f)
    f.close()

    train_seq = []
    train_label = []
    train_fname = []

    print(str(len(folds))+' folds found in '+path)

    for i in range(len(folds)):
        if not i == test_id:
            # folds for training
            seqs, labels, fnames = folds[i]['samples'],folds[i]['labels'], folds[i]['filenames']
            print 'train fold',i,'size= ',len(seqs)
            train_seq.extend(seqs)
            train_label.extend(labels)
            train_fname.extend(fnames)
        else:
            # the only fold for test
            print '[ test fold', i, 'size= ', len(folds[i]['samples']), "]"

    train_size = int(len(train_seq)*(1.-valid_portion))
    print('train size:'+str(train_size))
    valid_idx = range(len(train_seq))

    np.random.shuffle(valid_idx)

    valid_seq = [train_seq[s] for s in valid_idx[train_size:]]
    valid_label = [train_label[s] for s in valid_idx[train_size:]]
    valid_fname = [train_fname[s] for s in valid_idx[train_size:]]

    train_seq = [train_seq[s] for s in valid_idx[:train_size]]
    train_label = [train_label[s] for s in valid_idx[:train_size]]
    train_fname = [train_fname[s] for s in valid_idx[:train_size]]

    test_seq = folds[test_id]['samples']
    test_label = folds[test_id]['labels']
    test_fname = folds[test_id]['filenames']

    if sort_by_len:
        sorted_index = len_argsort(test_seq)
        test_seq = [test_seq[i] for i in sorted_index]
        test_label = [test_label[i] for i in sorted_index]
        test_fname = [test_fname[i] for i in sorted_index]

        sorted_index = len_argsort(valid_seq)
        valid_seq = [valid_seq[i] for i in sorted_index]
        valid_label = [valid_label[i] for i in sorted_index]
        valid_fname = [valid_fname[i] for i in sorted_index]

        sorted_index = len_argsort(train_seq)
        train_seq = [train_seq[i] for i in sorted_index]
        train_label = [train_label[i] for i in sorted_index]
        train_fname = [train_fname[i] for i in sorted_index]

    train = (train_seq, train_label, train_fname)
    valid = (valid_seq, valid_label, valid_fname)
    test = (test_seq, test_label, test_fname)

    assert not Has(train[2], test[2])
    assert not Has(train[2], valid[2])

    # train[0] is seqs, train[1] is labels, train[2] is filenames
    return train, valid, test


def prepare_data(seqs, labels, maxlen):
    batch_size = len(seqs)
    xdim = seqs[0].shape[1]

    x = np.zeros((batch_size, 1, maxlen, xdim)).astype(theano.config.floatX)
    for i in range(batch_size):
        # do padding for each sequence
        x[i, 0] = seq_padding(seqs[i], maxlen)

    y = np.array(labels)

    return x, y


def prepare_data_by_class(seqs, labels):
    assert len(seqs) == len(labels)
    res = OrderedDict()
    for i in range(len(seqs)):
        label = labels[i]
        if res.has_key(label):
            res[label].append(seqs[i])
        else:
            res[label] = [seqs[i]]

    sum = 0
    for k, v in res.items():
        res[k] = [v, [k for _ in range(len(v))]]
        sum += len(v)
    assert sum == len(seqs)

    return res


def build_network(options):

    input_var = T.tensor4('input_var', dtype=theano.config.floatX)
    target_var = T.vector('target_var', dtype='int64')

    ## -----------------------------Network Defination----------------------------- ##
    # do noting, no network parameters
    # shape is [batch_size, channel_num, max_seq_length, seq_dimension]
    network = lasagne.layers.InputLayer(shape=[None, 1, options['maxlen'], options['xdim']],
                                        input_var=input_var)
    # 1st convolution with 18 10x2 kernels
    network = lasagne.layers.Conv2DLayer(incoming=network, num_filters=18,
                                         filter_size=(10, 2))
    # Max pooling with 2x1 size
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 1))

    # 2nd convolution with 36 10x1 kernels
    network = lasagne.layers.Conv2DLayer(incoming=network, num_filters=36,
                                         filter_size=(10, 1))

    # Max pooling with 2x1 size
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 1))

    # 3rd convolution with 24 10x1 kernels
    network = lasagne.layers.Conv2DLayer(incoming=network, num_filters=24,
                                         filter_size=(10, 1))

    # Max pooling with 2x1 size
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 1))

    # input dropout by 50%, then fully connected of 256 neurons, f(.) = relu
    network = lasagne.layers.DenseLayer(incoming=lasagne.layers.dropout(network, p=0.5),
                                        num_units=64,
                                        nonlinearity=lasagne.nonlinearities.rectify)

    # input dropout by 50%, then softmax, 12 classes
    network = lasagne.layers.DenseLayer(incoming=lasagne.layers.dropout(network, p=0.5),
                                        num_units=options['ydim'],
                                        nonlinearity=lasagne.nonlinearities.softmax)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=0.01)

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),target_var),
                      dtype=theano.config.floatX)

    # compile functions
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    f_pred = theano.function([input_var], prediction.argmax(axis=1), name="f_pred")
    f_pred_val = theano.function([input_var], test_prediction.argmax(axis=1), name='f_pred_test')

    return network, train_fn, val_fn, f_pred, f_pred_val


def pred_err(data, f_pred_test, iterator, maxlen):
    # data[0]:seqs, [1]:labels
    Correct = 0
    for _, valid_index in iterator:
        x, y = prepare_data([data[0][t] for t in valid_index], [data[1][t] for t in valid_index], maxlen)
        preds = f_pred_test(x)  # the SoftMax output (probabilities)
        Correct += (preds == y).sum()

    correct_err = np.asarray(Correct, dtype=theano.config.floatX) / len(data[0])
    print "correct =", Correct, "total =", len(data[0])

    # get error rate
    return 1. - correct_err


def cnn_train(
    dataset="WatchDB_mixed_users_advanced_5folds.pkl",
    saveto="WatchDB_12class_CNN.npz",
    reload_model=None,
    max_epoches=5000,
    xdim=6,
    batch_size=64,
    ydim=12,
    display_Freq=10,
    save_Freq=5000,
    patience=15,
    valid_Freq=100,
    fold_id=0,
    maxlen=0,
):
    print "Training fold No.", fold_id
    model_options = locals().copy()

    # load data into 3 sets
    train, valid, test = load_data_N_folds(path=dataset, default_path=dataset, test_id=fold_id, valid_portion=0.1)

    # decide the maxlen
    model_options['maxlen'] = np.max(np.array([np.max(np.array([s.shape[0] for s in train[0]])),
                                               np.max(np.array([s.shape[0] for s in valid[0]])),
                                               np.max(np.array([s.shape[0] for s in test[0]]))]))
    print "The longest length is", model_options['maxlen']

    model_options["saveto"] = saveto[:-4]+"_fold_"+str(fold_id)+saveto[-4:]

    # check whether ydim is correct
    L = []
    for lb in train[1]:
        if lb not in L: L.append(lb)
    assert ydim == len(L)

    assert model_options['maxlen'] != 0
    # build network and compile functions
    network, f_train, f_val, f_pred, f_pred_test = build_network(model_options)

    kf_valid = get_minibatches_idx(len(valid[0]), batch_size, shuffle=False)
    kf_test = get_minibatches_idx(len(test[0]), batch_size, shuffle=False)

    history_errs = []
    best_p = None
    uidx = 0  # the number of update done
    estop = False  # early stop
    sub_err = OrderedDict();
    start_time = time.time()

    try:
        print
        for eidx in range(max_epoches):
            n_sample = 0
            kf_train = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_idx in kf_train:
                uidx += 1
                # build a mini-batch
                x, y = prepare_data([train[0][i] for i in train_idx],
                                    [train[1][i] for i in train_idx],
                                    model_options['maxlen'])

                n_sample += x.shape[0]  # actually is batchsize
                cost = f_train(x,y)

                if np.isnan(cost) or np.isinf(cost):
                    print('Bad Cost Detected: ', cost)
                    return 1., 1., 1.
                if np.mod(uidx, display_Freq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if np.mod(uidx, save_Freq) == 0:
                    print "Saving model into:", model_options["saveto"], "..."
                    if model_options["saveto"] is not None:
                        np.savez(model_options["saveto"], *lasagne.layers.get_all_param_values(network))

                if np.mod(uidx, valid_Freq) == 0:
                    train_err = pred_err(train, f_pred, kf_train, model_options['maxlen'])
                    valid_err = pred_err(valid, f_pred_test, kf_valid, model_options['maxlen'])
                    test_err = pred_err(test, f_pred_test, kf_test, model_options['maxlen'])
                    sub_err = f_subtest(test, f_pred_test,model_options['maxlen'])
                    history_errs.append([valid_err, test_err])

                    if (best_p is None or valid_err <= np.array(history_errs)[:,0].min()):
                        best_p = lasagne.layers.get_all_param_values(network)
                        bad_counter = 0
                    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err, 'patience ', bad_counter
                    if (len(history_errs) > patience and
                        valid_err >= np.array(history_errs)[:-patience, 0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            estop = True
                            break
            print "Seen", n_sample, "in", len(train[0]), "(total)"
            if estop:
                print('Early Stop!')
                break
    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    print 'The code run for %d epochs, with %f sec/epochs' % ((eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))

    if best_p is None:
        best_p = lasagne.layers.get_all_param_values(network)
    if model_options["saveto"] is not None:
        np.savez(model_options["saveto"], *best_p)

    # set best network parameters to the CNN
    lasagne.layers.set_all_param_values(network, best_p)
    train_err = pred_err(train, f_pred, kf_train, model_options["maxlen"])
    valid_err = pred_err(valid, f_pred_test, kf_valid, model_options["maxlen"])
    test_err = pred_err(test, f_pred_test, kf_test, model_options["maxlen"])

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    return train_err, valid_err, test_err

def f_subtest(test, f_pred_test , maxlen):
    # test_err = pred_err(test, f_pred_test, kf_test, maxlen)
    data_by_class = prepare_data_by_class(test[0], test[1])
    sub_err = OrderedDict()
    for k, v in data_by_class.items():
        x, y = prepare_data(v[0],v[1],maxlen)
        pred = f_pred_test(x)
        Correct = (pred == y).sum()
        sub_err[k] = 1.0 - Correct/float(len(v[0]))

    sub_err=OrderedDict(sorted(sub_err.items(), key=lambda t: t[0]))
    for k, v in sub_err.items():
        print k, v

    return sub_err


if __name__ == '__main__':
    # Errs = np.zeros((3, 5, 3))
    #
    # for i in range(5):
    #     Errs[0, i] = cnn_train(dataset="DB1_mixed_users_5folds.pkl", saveto="DB1_cnn.npz", xdim=3, ydim=10, valid_Freq=180, patience=8, fold_id=i)
    # for i in range(5):
    #     Errs[1, i] = cnn_train(dataset="DB2_mixed_users_5folds.pkl", saveto="DB2_cnn.npz", xdim=6, ydim=36, fold_id=i)
    # for i in range(5):
    #     Errs[2, i] = cnn_train(dataset="awdb_mixed_users_5folds.pkl", saveto="DB3_cnn.npz", xdim=6, ydim=62, fold_id=i)
    #
    # toSave = []
    # for i in range(3):
    #     toSave.append([Errs[i, j, 2] for j in range(5)])
    # np.savetxt("CNN_Errs.txt", np.array(toSave))
    #
    # for i in range(3):
    #     print "DB" + str(i)
    #     for j in range(5):
    #         print "fold", str(j), "train_err =", str(Errs[i, j, 0]), "valid_err =", str(Errs[i, j, 1]), "test_err =", str(
    #             Errs[i, j, 2])

    Errs = np.zeros((2, 5, 3))

    for i in range(5):
        Errs[0, i] = cnn_train(dataset="WatchDB_5class_advanced_5folds.pkl", saveto="Watch_CNN_5class.npz", xdim=6,
                                ydim=5, patience=10,
                                fold_id=i)

    for i in range(5):
        Errs[1, i] = cnn_train(dataset="WatchDB_mixed_users_advanced_5folds.pkl", saveto="Watch_CNN_12class.npz",
                                xdim=6, ydim=12,
                                fold_id=i)

    toSave = []
    for i in range(2):
        toSave.append([Errs[i, j, 2] for j in range(5)])
    np.savetxt("CNN_Watch_Errs.txt", np.array(toSave))

    for i in range(2):
        if (i == 0):
            print "Watch_DB_5class"
        if (i == 1):
            print "Watch_DB_12class"

        for j in range(5):
            print "fold", str(j), "train_err =", str(Errs[i, j, 0]), "valid_err =", str(
                Errs[i, j, 1]), "test_err =", str(Errs[i, j, 2])

    # for i in range(5):
    #     cnn_train(dataset="WatchDB_mixed_users_advanced_5folds.pkl", saveto="Watch_cnn_12class.npz", ydim=12, fold_id=i)
    # for i in range(5):
    #     cnn_train(dataset="WatchDB_5class_advanced_5folds.pkl", saveto="Watch_cnn_5class.npz", ydim=5, fold_id=i)