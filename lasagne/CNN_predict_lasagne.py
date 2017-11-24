import theano.tensor as T
import theano
import lasagne
import os
import gzip
import cPickle as pickle
from collections import OrderedDict
import numpy as np

def load_data_N_folds(path, default_path, test_id = 0, sort_by_len=True):
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

    test_prediction = lasagne.layers.get_output(network, deterministic=True)


    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),target_var),
                      dtype=theano.config.floatX)

    # compile functions

    f_pred_prod = theano.function([input_var], test_prediction, name="f_pred_prod")
    f_pred_val = theano.function([input_var], test_prediction.argmax(axis=1), name='f_pred_test')

    return network, f_pred_prod, f_pred_val


def forward(
        dataset = 'DB1_mixed_users_5folds.pkl',
        xdim = 3,
        ydim = 10,
        maxlen = 0,
):
    model_options = locals().copy()