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
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class ConcurrentIter(mx.io.DataIter):
    def  __init__(self, count, batch_size, data):
        super(ConcurrentIter, self).__init__()
        self.data_source =data
        self.batch_size = batch_size
        self.count = count

    def __iter__(self):
        for i in range(self.count/self.batch_size):
            data = []
            label = []
            for j in range(self.batch_size):
                [new_data, new_label] = self.data[i*self.batch_size + j]
                data.append(new_data)
                label.append(new_label)

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_names = ['data']
            label_names = ['softmax_label']
            data_batch =Batch(data_names, data_all, label_names, label_all)
            yield data_batch

if __name__ == '__main__':

    pass