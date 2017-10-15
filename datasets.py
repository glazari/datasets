import numpy as np
import scipy.io
import os
import urllib
import tarfile

import matplotlib.pyplot as plt

main_folder = os.path.expanduser('~')+'/DataSets/'


def _make(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

def create_1_hot(y, num_classes=10):
    assert 1 in y.shape or len(y.shape) < 2
    n = len(y)
    y_1_hot = np.zeros((n,num_classes))
    y_1_hot[np.arange(n),y.flatten()] = 1
    
    return y_1_hot

def load_SVHN(folder='SVHN/'):
    folder = main_folder+folder
    
    if not os.path.isdir(folder):
        _download_SVHN()
    
    print('loading SVHN training images...')
    train_name = folder+'train_32x32.mat'
    mat = scipy.io.loadmat(train_name)
    train_x = mat['X'].transpose([3,0,1,2])
    y = mat['y']
    y[y==10] = 0
    train_y = create_1_hot(y)
    
    print('loading SVHN test images...') 
    test_name = folder+'test_32x32.mat'
    mat = scipy.io.loadmat(test_name)
    test_x = mat['X'].transpose([3,0,1,2])
    y = mat['y']
    y[y==10] = 0
    test_y = create_1_hot(y)
    
    
    train_set = (train_x, train_y)
    test_set = (test_x, test_y)
    
    return train_set, test_set

def _download_SVHN():
    
    _make(main_folder)
    
    folder = main_folder+'SVHN/'
    
    print('downloading SVHN... (235Mb This may take a while)')
    
    os.mkdir(folder)
    
    print('downloading trainset....')
    download_link = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
    f, m = urllib.request.urlretrieve(download_link, folder+'train_32x32.mat')
    
    print('downloading testset....')
    download_link = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    f, m = urllib.request.urlretrieve(download_link, folder+'test_32x32.mat')

def load_Cifar_10(folder='cifar-10-batches-py/'):
    folder = main_folder+folder
    
    if not os.path.isdir(folder):
        _download_Cifar_10()
    
    files_cifar = os.listdir(folder)
    data_files = [x for x in files_cifar if 'data' in x]
    test_file = [x for x in files_cifar if 'test' in  x][0]

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    batches = []
    images = []
    labels = []
    print('loading Cifar_10 training images...')
    for file in data_files:
        b = unpickle(folder+file)
        batches.append(b)
        images.append(b[b'data'])
        labels.append(b[b'labels'])

    training_images = np.concatenate(images)
    training_labels = np.concatenate(labels)
    y_1_hot = create_1_hot(training_labels)
    train = (training_images, y_1_hot) 
    
    print('loading Cifar_10 test images...')    
    test = unpickle(folder+test_file)
    test_images = test[b'data']
    test_labels = np.array(test[b'labels'])
    y_1_hot = create_1_hot(test_labels)
    test = (test_images, y_1_hot)
    
    return train, test

def _download_Cifar_10():
    
    _make(main_folder)
    
    print('downloading Cifar_10... (167Mb this may take a while)')
    download_file = main_folder + 'cifar-10-python.tar.gz'
    download_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    f, m = urllib.request.urlretrieve(download_link, download_file)
    
    print('extracting files to "%s"' % main_folder)
    tar = tarfile.open(f, "r:gz")
    tar.extractall(main_folder)
    tar.close()
    
    os.remove(download_file)

def load_MNIST(folder='MNIST/'):
    folder = main_folder+folder    
    
    #Tensor flow has a nice API for downloading mnist. In the future I will 
    #use an aproach that does not rely on tf.
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    
    #this function already downloads the files if they are not present
    mnist = input_data.read_data_sets(folder, one_hot=True)
    train_set = (mnist.train.images, mnist.train.labels)
    test_set = (mnist.test.images, mnist.test.labels)
    
    return (train_set, test_set)

    
def _download_MNIST(folder='MNIST/'):
    folder = main_folder+folder
    
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    _ = input_data.read_data_sets(folder, one_hot=True)

def batches(X,y,batch_size=128):
    assert len(X) == len(y)
    n = len(X)
    p = np.random.permutation(n)
    
    num_batches = n // batch_size
    for i in range(num_batches):
        start = i*batch_size
        end = start+batch_size
        yield X[p[start:end]], y[p[start:end]]

    left_over = n % batch_size
    yield X[p[-left_over:]], y[p[-left_over:]]

