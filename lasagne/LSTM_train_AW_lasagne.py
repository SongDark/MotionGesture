import theano
import theano.tensor as T
import lasagne
import numpy as np
import gzip
import time
import cPickle as pickle
import os
from collections import OrderedDict


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def prepare_data(seqs, labels):
    lengths = [np.shape(s)[0] for s in seqs]

    n_samples = np.shape(seqs)[0]
    maxlen = np.max(lengths)
    xdim = np.shape(seqs[0])[1]

    x = np.zeros((n_samples, maxlen, xdim)).astype(theano.config.floatX)
    x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)

    for n_idx in range(n_samples):
        for l_idx in range(np.shape(seqs[n_idx])[0]):
            x[n_idx][l_idx] = seqs[n_idx][l_idx]

    for idx, s in enumerate(seqs):
        x_mask[idx, :lengths[idx]] = 1.

    labels = np.array(labels)

    return x, x_mask, labels

def load_data_N_folds_smart(path,default_path, test_id = 0, start_ges = 0, end_ges = 62, valid_portion=0.1, sort_by_len=True):
    def get_dataset_file(dataset, default_dataset):
        data_dir, data_file = os.path.split(dataset)
        if data_dir == "" and not os.path.isfile(dataset):
            # Check if dataset is in the data directory.
            new_path = os.path.join(
                os.path.split(__file__)[0],
                "data",
                dataset
            )
            if os.path.isfile(new_path) or data_file == default_dataset:
                dataset = new_path
        return dataset

    path = get_dataset_file(path,default_path)
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

    seqs, labels, fnames = folds[test_id]['samples'],folds[test_id]['labels'], folds[test_id]['filenames']
    [test_seq, test_label, test_fname]=[[],[],[]]
    for sample_i in range(len(seqs)):
        if(labels[sample_i]>=start_ges and labels[sample_i]<=end_ges):
            test_seq.append(seqs[sample_i]);
            test_label.append(labels[sample_i]-start_ges)
            test_fname.append(fnames[sample_i])

    for i in range(len(folds)):
        if not i == test_id:
            seqs, labels, fnames = folds[i]['samples'],folds[i]['labels'], folds[i]['filenames']
            sample_size = 0;
            for sample_i in range(len(seqs)):
                if(labels[sample_i]>=start_ges and labels[sample_i]<=end_ges):
                    train_seq.append(seqs[sample_i])
                    train_label.append(labels[sample_i]-start_ges)
                    train_fname.append(fnames[sample_i])
                    sample_size += 1;
            print 'i=' + str(i), 'samples=' + str(sample_size)
        else:
            print 'i=' + str(i), 'test_id=' + str(test_id), 'samples=' + str(len(test_seq))

    train_size = int(len(train_seq)*(1.-valid_portion))
    valid_idx = range(len(train_seq))
    np.random.shuffle(valid_idx)
    valid_seq = [train_seq[s] for s in valid_idx[train_size:]]
    valid_label = [train_label[s] for s in valid_idx[train_size:]]
    valid_fname = [train_fname[s] for s in valid_idx[train_size:]]
    train_seq = [train_seq[s] for s in valid_idx[:train_size]]
    train_label = [train_label[s] for s in valid_idx[:train_size]]
    train_fname = [train_fname[s] for s in valid_idx[:train_size]]

    print
    print "train size =", len(train_seq)
    print "valid size =", len(valid_seq)
    print "test size =", len(test_seq)
    print

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

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

    return train,valid,test

def load_data_N_folds(path,default_path, test_id = 0, valid_portion=0.1, sort_by_len=True):
    def get_dataset_file(dataset, default_dataset):
        data_dir, data_file = os.path.split(dataset)
        if data_dir == "" and not os.path.isfile(dataset):
            # Check if dataset is in the data directory.
            new_path = os.path.join(
                os.path.split(__file__)[0],
                "data",
                dataset
            )
            if os.path.isfile(new_path) or data_file == default_dataset:
                dataset = new_path
        return dataset

    path = get_dataset_file(path,default_path)
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
            seqs, labels, fnames = folds[i]['samples'],folds[i]['labels'], folds[i]['filenames']
            train_seq.extend(seqs)
            train_label.extend(labels)
            train_fname.extend(fnames)
            print 'i=' + str(i), 'samples=' + str(len(folds[i]['samples']))
        else:
            print 'i=' + str(i), 'test_id=' + str(test_id), 'samples=' + str(len(folds[i]['samples']))

    train_size = int(len(train_seq)*(1.-valid_portion))
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

    print
    print "train size =", len(train_seq)
    print "valid size =", len(valid_seq)
    print "test size =", len(test_seq)
    print

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

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

    return train,valid,test


def build_network(options):

    input_var = T.tensor3('input_var', dtype=theano.config.floatX)
    mask_var = T.matrix('mask_var', dtype=theano.config.floatX)
    target_var = T.vector('target_var', dtype='int64')

    ## -----------------------------Network Defination----------------------------- ##
    # do noting, no network parameters
    l_input = lasagne.layers.InputLayer(shape=[None, None, options['xdim']],
                                        input_var=input_var)
    l_mask = lasagne.layers.InputLayer(shape=[None, None],
                                       input_var=mask_var)
    # full-connected input layer, out:[batch_size, max_len, num_hidden_unit]
    l_input_fc = lasagne.layers.DenseLayer_RNN(incoming=l_input,
                                               num_units=options['num_hidden_unit'])
    # lstm layer, return all outputs, out:[batch_size, max_len, num_hidden_unit]
    l_lstm = lasagne.layers.LSTMLayer(incoming=l_input_fc,
                                      num_units=options['num_hidden_unit'],
                                      grad_clipping=100,
                                      mask_input=l_mask,
                                      only_return_final=False)
    # mean pooling through the time, out:[batch_size, num_hidden_unit]
    l_mean_pool = lasagne.layers.MeanPoolLayer_Rnn(incoming=l_lstm,
                                                   mask_input=l_mask)

    # dropout to avoid over-fitting, p is drop rate, out:[batch_size, num_hidden_unit]
    l_dropout = lasagne.layers.DropoutLayer(incoming=l_mean_pool,
                                            p=0.5 if options['dropout'] else 0)
    # output layer, using softmax, out:[batch_size, ydim]
    l_out = lasagne.layers.DenseLayer(incoming=l_dropout,
                                      num_units=options['ydim'],
                                      nonlinearity=lasagne.nonlinearities.softmax)
    # turn Layer instance to TensorVariable
    pred = lasagne.layers.get_output(l_out)
    # this abandons the dropout layer
    pred_test = lasagne.layers.get_output(l_out, deterministic=True)

    # loss function, using cross entropy
    # cost = -T.log(pred[T.arange(options['batch_size']), target_var]).mean()
    cost = T.nnet.categorical_crossentropy(pred, target_var).mean()

    # obtain parameters of the whole network
    all_params = lasagne.layers.get_all_params(l_out)

    ## ----------------------------Function Defination---------------------------- ##
    # in adadelta, initial learning rate is not needed
    updates = lasagne.updates.adadelta(cost, all_params)

    print "Compiling functions ..."
    # [batch_size, ydim], the softmax output
    f_pred_prob = theano.function([input_var, mask_var], pred, name='f_pred_prob')
    f_pred = theano.function([input_var, mask_var], pred.argmax(axis=1), name='f_pred')  # [batch_size, 1]
    f_pred_test = theano.function([input_var, mask_var], pred_test.argmax(axis=1), name='f_pred_test')
    f_cost = theano.function([input_var, mask_var, target_var], cost, name='f_cost')  # scalar

    # compute cost and upgrade params
    f_train = theano.function([input_var, mask_var, target_var], cost, updates=updates, name='f_train')

    return l_out, f_pred_prob, f_pred, f_pred_test, f_cost, f_train


def pred_err(data, f_pred_test, iterator):
    # data[0]:seqs, [1]:labels
    Correct = 0
    for _, valid_index in iterator:
        x = [data[0][t] for t in valid_index]
        y = [data[1][t] for t in valid_index]
        x, mask, y = prepare_data(x,y)
        preds = f_pred_test(x, mask)
        Correct += (preds == y).sum()

    correct_err = np.asarray(Correct, dtype=theano.config.floatX) / len(data[0])
    print "correct =", Correct
    print "total =", len(data[0])
    print "accuracy =", correct_err
    return 1. - correct_err


def lstm_train(
        dataset=None,
        saveto=None,
        reload_model=None,
        max_epochs=1000,
        batch_size=64,
        xdim=6,
        num_hidden_unit=-1,
        ydim=-1,
        dropout=True,
        patience=15,
        display_freq=10,
        valid_freq=350,
        save_freq=1200,
        fold_id=-1,
        start_ges = 0,
        end_ges=61
):
    network_options = locals().copy()
    network_options["saveto"] = saveto[:-4] + "_fold_" + str(fold_id) + saveto[-4:]

    print "In", fold_id, "th, loading data"
    train, valid, test = load_data_N_folds_smart(path=dataset, default_path=dataset, test_id=fold_id, start_ges=start_ges, end_ges=end_ges, valid_portion=0.1)

    L = []
    for lb in train[1]:
        if lb not in L: L.append(lb)

    print ydim,len(L)
    assert ydim == len(L)

    print "Building Network ..."
    network, f_pred_prob, f_pred, f_pred_test, f_cost, f_train=build_network(network_options)

    if reload_model is not None:
        print 'Re-loading model from ', reload_model, '...'
        with np.load(reload_model) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    #
    kf_valid = get_minibatches_idx(len(valid[0]), batch_size, shuffle=False)
    kf_test = get_minibatches_idx(len(test[0]), batch_size, shuffle=False)

    history_errs = []
    best_p = None
    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()

    historical_train_errors=[]
    historical_valid_errors=[]
    historical_test_errors=[]
    historical_Cost = []

    try:
        for eidx in range(max_epochs):
            n_sample = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1

                # Select the random examples for this minibatch
                x = [train[0][t] for t in train_index]
                y = [train[1][t] for t in train_index]
                x, mask, y = prepare_data(x, y)

                n_sample += x.shape[0]
                cost = f_train(x, mask, y)

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if np.mod(uidx, display_freq) == 0:
                    historical_Cost.append(cost)
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if np.mod(uidx, save_freq) == 0:
                    print "Saving model ... "
                    if network_options["saveto"] is not None:
                        np.savez(network_options["saveto"], *lasagne.layers.get_all_param_values(network))

                if np.mod(uidx, valid_freq) == 0:
                    train_err = pred_err(train, f_pred_test, kf)
                    valid_err = pred_err(valid, f_pred_test, kf_valid)
                    test_err = pred_err(test, f_pred_test, kf_test)

                    historical_train_errors.append(train_err)
                    historical_valid_errors.append(valid_err)
                    historical_test_errors.append(test_err)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= np.array(history_errs)[:,0].min()):
                        best_p = lasagne.layers.get_all_param_values(network)
                        bad_counter = 0
                    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err, 'patience ', bad_counter
                    if (len(history_errs) > patience and
                        valid_err >= np.array(history_errs)[:-patience,0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_sample)
            if estop:
                break



    except KeyboardInterrupt:
        print("Training interupted")

    E = np.zeros((3,len(historical_train_errors)))
    E[0,:] = historical_train_errors
    E[1,:] = historical_valid_errors
    E[2,:] = historical_test_errors
    print E
    np.savetxt(dataset+'historical_Err_'+str(num_hidden_unit)+'.txt',E)
    np.savetxt(dataset+'historical_cost_'+str(num_hidden_unit)+'.txt',historical_Cost)

    end_time = time.time()
    print 'The code run for %d epochs, with %f sec/epochs' % ((eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))

    if best_p is None:
        best_p = lasagne.layers.get_all_param_values(network)
    if network_options["saveto"] is not None:
        np.savez(network_options["saveto"], *best_p)

    lasagne.layers.set_all_param_values(network, best_p)
    train_err = pred_err(train, f_pred_test, kf)
    valid_err = pred_err(valid, f_pred_test, kf_valid)
    test_err = pred_err(test, f_pred_test, kf_test)
    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    return train_err, valid_err, test_err

def f_subtest(test, f_pred_test , maxlen):
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
    data_by_class = prepare_data_by_class(test[0], test[1])
    sub_err = OrderedDict()
    for k, v in data_by_class.items():
        x, y = prepare_data(v[0],v[1],maxlen)
        pred = f_pred_test(x)
        Correct = (pred == y).sum()
        sub_err[k] = 1.0 - Correct/float(len(v[0]))

    sub_err=OrderedDict(sorted(sub_err.items(), key=lambda t: t[0]))
    for k, v in sub_err.items():
        print "label", k, "error =", v

    return sub_err


if __name__ == '__main__':
    print
    # A-Z
    folds_N = 1
    Errs_0 = np.zeros((1, folds_N, 3))
    # for i in range(5):
    #     Errs_0[0, i] = lstm_train(dataset="DB2_userindependent_2_16_5folds.pkl", saveto="DB2_lstm_144_Upper_independent.npz", xdim=6, ydim=26, num_hidden_unit=144,
    #                              fold_id=i, start_ges=10, end_ges=35)

    i = 0
    Errs_0[0, i] = lstm_train(dataset="DB1_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=3, ydim=10,
                              num_hidden_unit=256,
                              patience=15, valid_freq=350,
                              fold_id=i, start_ges=0, end_ges=10)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="awdb_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=62, num_hidden_unit=16,
    #                               patience=15, valid_freq=350,
    #                              fold_id=i, start_ges=0, end_ges=62) #a-z
    # Errs_0[0, i] = lstm_train(dataset="awdb_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=62,
    #                           num_hidden_unit=32,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=62)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="awdb_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=62,
    #                           num_hidden_unit=64,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=62)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="awdb_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=62,
    #                           num_hidden_unit=96,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=62)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="awdb_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=62,
    #                           num_hidden_unit=128,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=62)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="awdb_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=62,
    #                           num_hidden_unit=144,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=62)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="awdb_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=62,
    #                           num_hidden_unit=186,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=62)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="awdb_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=62,
    #                           num_hidden_unit=256,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=62)  # a-z
    #
    #
    # Errs_0[0, i] = lstm_train(dataset="DB2_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=36,
    #                           num_hidden_unit=16,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=36)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="DB2_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=36,
    #                           num_hidden_unit=32,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=36)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="DB2_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=36,
    #                           num_hidden_unit=64,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=36)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="DB2_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=36,
    #                           num_hidden_unit=96,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=36)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="DB2_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=36,
    #                           num_hidden_unit=128,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=36)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="DB2_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=36,
    #                           num_hidden_unit=144,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=36)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="DB2_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=36,
    #                           num_hidden_unit=186,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=36)  # a-z
    # Errs_0[0, i] = lstm_train(dataset="DB2_mixed_users_5folds.pkl", saveto="nothing.npz", xdim=6, ydim=36,
    #                           num_hidden_unit=256,
    #                           patience=15, valid_freq=350,
    #                           fold_id=i, start_ges=0, end_ges=36)  # a-z



    # toSave = []
    # for i in range(2):
    #     toSave.append([Errs_1[i, j, 2] for j in range(5)])
    # np.savetxt("Lstm_Errs_144_Upper.txt", np.array(toSave))
    #
    # Errs_2 = np.zeros((1, 5, 3))
    # for i in range(5):
    #     Errs_2[0, i] = lstm_train(dataset="awdb_mixed_users_5folds.pkl", saveto="DB3_lstm_144_lower.npz", xdim=6,
    #                               ydim=26, num_hidden_unit=144,
    #                               fold_id=i, start_ges=10, end_ges=35)
    # toSave = []
    # for i in range(1):
    #     toSave.append([Errs_2[i, j, 2] for j in range(5)])
    # np.savetxt("Lstm_Errs_144_lower.txt", np.array(toSave))
    #
    # print "144 upper"
    # for i in range(2):
    #     print "DB"+str(i+2)
    #     for j in range(5):
    #         print "fold", str(j), "train_err =", str(Errs_1[i,j,0]), "valid_err =", str(Errs_1[i,j,1]), "test_err =", str(Errs_1[i,j,2])
    #
    # print "144 lower"
    # for i in range(1):
    #     print "DB" + str(i + 3)
    #     for j in range(5):
    #         print "fold", str(j), "train_err =", str(Errs_2[i, j, 0]), "valid_err =", str(Errs_2[i, j, 1]), "test_err =", str(Errs_2[i, j, 2])

    ##########################################################

    Errs_1 = np.zeros((1,5,3))

    # for i in range(5):
    #     Errs_1[0, i] = lstm_train(dataset="DB1_mixed_users_5folds.pkl", saveto="DB1_lstm_160.npz", xdim=3, ydim=10, num_hidden_unit=160,
    #                             valid_freq=180, patience=10, fold_id=i)
    # for i in range(5):
    #     Errs_1[1, i] = lstm_train(dataset="DB2_mixed_users_5folds.pkl", saveto="DB2_lstm_160.npz", xdim=6, ydim=36, num_hidden_unit=160,
    #                             fold_id=i)
    # for i in range(5):
    #     Errs_1[0, i] = lstm_train(dataset="DB1_userindependent_2_16_5folds.pkl", saveto="DB1_lstm_144_2_16_independent.npz", xdim=3, ydim=10, num_hidden_unit=144,
    #                               patience=12, start_ges=0, end_ges=9, valid_freq=300,
    #                               # reload_model='DB2_lstm_144_fold_0.npz',
    #                             fold_id=i)

    # toSave = []
    # for i in range(3):
    #     toSave.append([Errs_1[i, j, 2] for j in range(5)])
    # np.savetxt("Lstm_Errs_160.txt", np.array(toSave))
    #
    # Errs_2 = np.zeros((3, 5, 3))
    #
    # for i in range(5):
    #     Errs_2[0, i] = lstm_train(dataset="DB1_mixed_users_5folds.pkl", saveto="DB1_lstm_256.npz", xdim=3, ydim=10, num_hidden_unit=256,
    #                             valid_freq=180, patience=10, fold_id=i)
    # for i in range(5):
    #     Errs_2[1, i] = lstm_train(dataset="DB2_mixed_users_5folds.pkl", saveto="DB2_lstm_256.npz", xdim=6, ydim=36, num_hidden_unit=256,
    #                             fold_id=i)
    # for i in range(5):
    #     Errs_2[2, i] = lstm_train(dataset="awdb_mixed_users_5folds.pkl", saveto="DB3_lstm_256.npz", xdim=6, ydim=62, num_hidden_unit=256,
    #                             fold_id=i)
    #
    # toSave = []
    # for i in range(3):
    #     toSave.append([Errs_2[i,j,2] for j in range(5)])
    # np.savetxt("Lstm_Errs_256.txt", np.array(toSave))

    # Errs_3 = np.zeros((3, 5, 3))
    #
    # for i in range(5):
    #     Errs_3[0, i] = lstm_train(dataset="DB1_mixed_users_5folds.pkl", saveto="DB1_lstm_128.npz", xdim=3, ydim=10,
    #                               num_hidden_unit=128, valid_freq=180, patience=10, fold_id=i)
    # for i in range(5):
    #     Errs_3[1, i] = lstm_train(dataset="DB2_mixed_users_5folds.pkl", saveto="DB2_lstm_128.npz", xdim=6, ydim=36,
    #                               num_hidden_unit=128, fold_id=i)
    # for i in range(5):
    #     Errs_3[2, i] = lstm_train(dataset="awdb_mixed_users_5folds.pkl", saveto="DB3_lstm_128.npz", xdim=6, ydim=62,
    #                               num_hidden_unit=128, fold_id=i)
    #
    # toSave = []
    # for i in range(3):
    #     toSave.append([Errs_3[i, j, 2] for j in range(5)])
    # np.savetxt("Lstm_Errs_128.txt", np.array(toSave))

    # print "144"
    # for i in range(3):
    #     print "DB"+str(i)
    #     for j in range(5):
    #         print "fold", str(j), "train_err =", str(Errs_1[i,j,0]), "valid_err =", str(Errs_1[i,j,1]), "test_err =", str(Errs_1[i,j,2])
    # print "256"
    # for i in range(3):
    #     print "DB"+str(i)
    #     for j in range(5):
    #         print "fold", str(j), "train_err =", str(Errs_2[i,j,0]), "valid_err =", str(Errs_2[i,j,1]), "test_err =", str(Errs_2[i,j,2])
    # print "128"
    # for i in range(3):
    #     print "DB"+str(i)
    #     for j in range(5):
    #         print "fold", str(j), "train_err =", str(Errs_3[i,j,0]), "valid_err =", str(Errs_3[i,j,1]), "test_err =", str(Errs_3[i,j,2])

    # for i in range(1):
    #     if i == 0:
    #         print "DB2_lstm_144_Upper_independent.npz"
    #     if i == 1:
    #         print "DB3_lstm_144_lower_independent.npz"
    #
    #     for j in range(5):
    #         print "fold", str(j), "train_err =", str(Errs_0[i, j, 0]), "valid_err =", str(
    #             Errs_0[i, j, 1]), "test_err =", str(Errs_0[i, j, 2])

    for i in range(1):

        # print "DB3_lstm_144_2_16_num_independent.npz"

        for j in range(folds_N):
            print "fold", str(j), "train_err =", str(Errs_0[i,j,0]), "valid_err =", str(Errs_0[i,j,1]), "test_err =", str(Errs_0[i,j,2])


