from keras import backend as K
from keras.layers.convolutional import Convolution1D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.visualize_util import plot
from keras.utils import np_utils
from six.moves import cPickle as pickle
import numpy

def max_1d(X):
    return K.max(X, axis=1)

with open('data/input.pickle', 'rb') as f:
    save = pickle.load(f)
    train_features = save['train_data']
    test_features = save['test_data']
    test_labels = save['test_labels']
    train_labels = save['train_labels']
    del save  # gc

test_labels = np_utils.to_categorical(test_labels, 6)
train_labels = np_utils.to_categorical(train_labels, 6)

# set parameters:
max_features = 5000
maxlen = 169
batch_size = 16
embedding_dims = 169
nb_filter = 32
filter_length = 8
hidden_dims = 64
nb_epoch = 15

print(len(train_features), 'train sequences')
print(len(test_features), 'test sequences')

print('Build model...')

nn = Sequential()

nn.add(LSTM(32, input_shape=(10, 64)))

nn.compile(loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])

nn.fit(train_features, train_labels, batch_size=batch_size, nb_epoch=nb_epoch)

# plot(nn, to_file='output.png', show_shapes='true')

#
# nn.add(Embedding(max_features, embedding_dims, input_length=maxlen, dropout=0.2))
# nn.add(Convolution1D(nb_filter=32,
#                      filter_length=5,
#                      border_mode='same',
#                      activation='relu',
#                      subsample_length=1))
#
# nn.add(Convolution1D(nb_filter=32,
#                      filter_length=3,
#                      border_mode='valid',
#                      activation='relu',
#                      subsample_length=1))
#
#
# nn.add(Convolution1D(nb_filter=32,
#                      filter_length=3,
#                      border_mode='valid',
#                      activation='relu',
#                      subsample_length=1))
#
# nn.add(Lambda(max_1d, output_shape=(nb_filter,)))
#
# nn.add(Dense(hidden_dims))
# nn.add(Dropout(0.2))
# nn.add(Activation('relu'))
#
# nn.add(Dense(6))
# nn.add(Activation('relu'))

