import mxnet as mx
import numpy as np
import platform

def getGenerator(data_source):
    for i in data_source:
        yield i

    def reset(self):
        pass

def is_macosx():
    return platform.system() == "Darwin"

class Batch(object):
    def __init__(self, data_names, data, label_names = None, label = None):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.pad = 0

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

def RMSE(label, pred):
    print(label)
    print(pred)
    ret = 0.0
    n = 0.0
    if pred.shape[1] == 8:
        return None
    for k in range(pred.shape[0]):
        v1 = label[k]
        v2 = pred[k][0]
        ret += abs(v1 - v2) / v1
        n += 1.0
    return ret / n

class MxIter(mx.io.DataIter):
    """
    data's format: [[name, np.ndarray]]
    label's format: [[name, np.ndarray]]
    data shape of the provide_data(label):(data_size, data_point_size)
    """
    def  __init__(self, data, label, count, batch_size):
        super(MxIter, self).__init__()
        assert isinstance(data, list)
        if label:
            assert isinstance(label, list)
            for key,value in label:
                assert isinstance(value, np.ndarray)
        for key,value in data:
            assert isinstance(value, np.ndarray)

        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.count = count

        self.provide_data = []
        for key, value in self.data:
            self.provide_data += [(key, (batch_size, value.shape[1]))]

        if label:
            self.provide_label = []
            for key,value in self.label:
                self.provide_label += [(key, (batch_size, value.shape[1]))]

    def get_new_batch(self, new_list, batch_index):
        list_data_names = []
        list_data_all = []
        for key,value in new_list:
            list_data_names += [key]
            new_data_array= []
            for j in range(self.batch_size):
                new_data = value[batch_index*self.batch_size + j]
                new_data_array.append(new_data)
            list_data_all += [mx.nd.array(new_data_array)]
        return list_data_names, list_data_all


    def __iter__(self):
        for i in range(self.count/self.batch_size):
            data_names, data_all = self.get_new_batch(self.data, i)
            if self.label:
                label_names, label_all = self.get_new_batch(self.label, i)
                data_batch = Batch(data_names, data_all, label_names, label_all)
                yield data_batch
            else:
                data_batch = Batch(data_names, data_all, None, None)
                yield data_batch

    def reset(self):
        pass

class PredictIter(mx.io.DataIter):

    pass

