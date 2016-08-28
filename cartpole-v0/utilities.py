import mxnet as mx
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

class MxIter(mx.io.DataIter):
    def  __init__(self, count, batch_size, data, label):
        super(MxIter, self).__init__()
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.count = count

        if type(data[0]) is float:
            shape = 1
        else:
            shape = len(data[0])
        self.provide_data = [('data', (batch_size, shape))]

        if label:
            if type(label[0]) is float:
                shape = 1
            else:
                shape = len(label[0])

        self.provide_label = [('label', (batch_size, shape))]

    def __iter__(self):
        for i in range(self.count/self.batch_size):
            data = []
            for j in range(self.batch_size):
                new_data = self.data[i*self.batch_size + j]
                data.append(new_data)
            data_all = [mx.nd.array(data)]
            data_names = ['data']

            if self.label:
                label = []
                for j in range(self.batch_size):
                    new_label = self.label[i*self.batch_size + j]
                    label.append(new_label)
                label_all = [mx.nd.array(label)]
                label_names = ['label']
                data_batch =Batch(data_names, data_all, label_names, label_all)
                yield data_batch
            else:
                data_batch =Batch(data_names, data_all, None, None)
                yield data_batch

    def reset(self):
        pass

class PredictIter(mx.io.DataIter):

    pass


if __name__ == '__main__':
    data = [[[1,2],[3,4]], [[1,2],[3,4]]]
    iter = MxIter(len(data), 1, data)
    for i in iter:
        print(i.provide_data)
    pass