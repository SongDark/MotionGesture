import theano.tensor as T
import theano
import lasagne
import os
import gzip
import cPickle as pickle
from collections import OrderedDict
import numpy as np

def prepare_data(seqs, labels, maxlen=None):
    lengths = [np.shape(s)[0] for s in seqs]
    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = np.shape(seqs)[0]
    maxlen = np.max(lengths)
    xdim = np.shape(seqs[0])[1]

    x = np.zeros((n_samples, maxlen, xdim)).astype(theano.config.floatX)
    x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)

    for n_idx in range(n_samples):
        for l_idx in range(np.shape(seqs[n_idx])[0]):
            x[n_idx][l_idx] = seqs[n_idx][l_idx]

    for idx, s in enumerate(seqs):
        # x[:lengths[idx], idx] = s
        x_mask[idx, :lengths[idx]] = 1.

    # # WatchDB: 7 class labels is 0~4, 12 and 13
    labels = np.array(labels)
    for i in range(len(labels)):
        if labels[i] == 12:
            labels[i] = 5
        else :
            if labels[i] == 13:
                labels[i] =6

    return x, x_mask, labels

def load_data_N_folds(path, default_path, test_id = 0, valid_portion=0.1, maxlen=None, sort_by_len=True):
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

    def Has(bigger_list, smaller_list):
        for string in smaller_list:
            if string in bigger_list:
                return True
        return False

    path = get_dataset_file(path,default_path)
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    folds = pickle.load(f)
    f.close()

    print(str(len(folds))+' folds found in '+path)

    for i in range(len(folds)):
        if not i == test_id:
            # folds for training
            seqs, labels, fnames = folds[i]['samples'],folds[i]['labels'], folds[i]['filenames']
            print('train fold',i,'size= ',len(seqs))
        else:
            # the only fold for test
            print('test fold', i, 'size= ', len(folds[i]['samples']))

    test_seq = folds[test_id]['samples']
    test_label = folds[test_id]['labels']
    test_fname = folds[test_id]['filenames']

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_seq)
        test_seq = [test_seq[i] for i in sorted_index]
        test_label = [test_label[i] for i in sorted_index]
        test_fname = [test_fname[i] for i in sorted_index]

    test = (test_seq, test_label, test_fname)

    return test

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

def pred_fuse(data, f_pred_test):
    x,mask,y=prepare_data(data[0],data[1])
    N = np.max(y)+1;
    fuse_matrix = np.zerso((N, N))
    preds = f_pred_test(x,mask)
    for i in range(len(y)):
        fuse_matrix[y[i], preds[i]] += 1


def pred_err(data, f_pred_test):
    # data[0]:seqs, [1]:labels
    Correct = 0
    x, mask, y = prepare_data(data[0], data[1])
    preds = f_pred_test(x, mask)
    Correct += (preds == y).sum()

    correct_rate = np.asarray(Correct, dtype=theano.config.floatX) / len(data[0])

    N = np.max(y) + 1;
    fuse_matrix = np.zeros((N, N))
    wrong_names=[]
    for i in range(len(y)):
        fuse_matrix[y[i], preds[i]] += 1
        if y[i]!=preds[i]:
            wrong_names.append(data[2][i])

    print "correct =", Correct
    print "total =", len(data[0])
    print 'err =', 1. - correct_rate
    print "accuracy =", correct_rate
    print fuse_matrix
    # for i in range(len(wrong_names)):
    #     print wrong_names[i]
    return 1. - correct_rate, correct_rate, fuse_matrix

def f_subtest(test, f_pred_test):

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
        x, mask, y = prepare_data(v[0],v[1])
        pred = f_pred_test(x, mask)
        Correct = (pred == y).sum()
        # print k, float(len(v[0]))
        sub_err[k] = 1.0 - Correct/float(len(v[0]))

    sub_err=OrderedDict(sorted(sub_err.items(), key=lambda t: t[0]))
    # for k, v in sub_err.items():
    #     print "label", k, "error =", v

    return sub_err

def forward(
        dataset="WatchDB_2_14_mix_horizontal_5folds.pkl",
        dropout=False,
        xdim=6,
        num_hidden_unit=64,
        ydim=7,
        fold_id=-1
):
    model_options = locals().copy()

    l_out, f_pred_prob, f_pred, f_pred_test, f_cost, f_train = build_network(model_options)

    err = [];
    cor = [];
    sub_err = OrderedDict()
    fuse_matrix = np.zeros((ydim,ydim))

    for fold_id in range(5):
        test = load_data_N_folds(path=dataset, default_path=dataset, test_id=fold_id, valid_portion=0.1)
        model_name = "WatchDB_2_14_new_horizontal_7class_64_fold_" + str(fold_id) + ".npz";
        with np.load(model_name) as f:
            model = [f['arr_%d' % i] for i in range(len(f.files))]

        len(model)
        lasagne.layers.set_all_param_values(l_out, model)

        e,c, fm = pred_err(test, f_pred_test)
        print fm.shape
        sub_err['fold'+str(fold_id)] = f_subtest(test, f_pred_test)

        err.append(e);
        cor.append(c);
        fuse_matrix = fuse_matrix + fm

    print err
    print np.mean(np.array(err))
    print cor
    print np.mean(np.array(cor))

    print
    for i in range(5):
        fold = sub_err['fold'+str(i)]
        print 'fold'+str(i)
        for k,v in fold.items():
            print k, v
    print
    print fuse_matrix
    # np.savetxt('DB3_144.txt', fuse_matrix.astype('int32'))
    print str(np.sum(np.diag(fuse_matrix))),"corrects in", str(np.sum(fuse_matrix)),",", str(np.sum(np.diag(fuse_matrix)/float(np.sum(fuse_matrix))))
    for i in range(np.size(fuse_matrix,1)):
        print "gesture",str(i),":",str(fuse_matrix[i,i]),"corrects in", str(np.sum(fuse_matrix[i])), ",", str(fuse_matrix[i,i]/float(np.sum(fuse_matrix[i])))

forward()