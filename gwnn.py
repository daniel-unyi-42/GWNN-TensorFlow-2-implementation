import tensorflow as tf
import scipy.integrate as integrate
from scipy import sparse
from scipy.sparse.linalg import eigsh
from numpy import exp, cos, pi

def extract_data(filename):
  import numpy as np
  # load data
  data = np.load(filename)
  adj = sparse.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']), \
                                       shape=data['adj_shape'], dtype='float32')
  attr = sparse.csr_matrix((data['attr_data'], data['attr_indices'], data['attr_indptr']), \
                                     shape=data['attr_shape'], dtype='float32')
  N = len(data['labels'])
  N_classes = len(set(data['labels']))
  labels = sparse.csr_matrix((np.ones(N), data['labels'], np.arange(N + 1)), \
                             shape=(N, N_classes), dtype='float32')
  # symmetrize
  adj = adj.tolil()
  rows, cols = adj.nonzero()
  adj[cols, rows] = adj[rows, cols]
  # shuffle
  permutation = list(range(N))
  #np.random.shuffle(permutation)
  adj = adj.tocoo()
  for i in range(adj.nnz):
    adj.row[i] = permutation[adj.row[i]]
    adj.col[i] = permutation[adj.col[i]]
  attr = attr.tocoo()
  for i in range(attr.nnz):
    attr.row[i] = permutation[attr.row[i]]
  labels = labels.tocoo()
  for i in range(labels.nnz):
    labels.row[i] = permutation[labels.row[i]]
  # result
  return adj.tocsr(), attr.todense(), labels.todense()

# hyperparameters
scale = 1.0
thresold = 1e-3
K = 4
hidden = 128
learning_rate = 0.01
epochs = 100
split = 542

filename = 'cora.npz'

adjacency_matrix, feature_matrix, label_matrix = extract_data(filename)

feature_matrix = tf.math.l2_normalize(tf.constant(feature_matrix), axis=1)
label_matrix = tf.math.l2_normalize(tf.constant(label_matrix), axis=1)

# train test split
N = label_matrix.shape[0]
train_mask = tf.concat((tf.ones((split, 1)), tf.zeros((N - split, 1))), 0)
train_matrix = train_mask * label_matrix
test_mask = tf.concat((tf.zeros((split, 1)), tf.ones((N - split, 1))), 0)
test_matrix = test_mask * label_matrix

# renormalization
degrees = sparse.csr_matrix(adjacency_matrix.sum(axis=1))
norm = degrees.power(-0.5)
laplacian = sparse.eye(N, dtype='float32') - adjacency_matrix.multiply(norm).T.multiply(norm)  # I - D**-1/2 * W * D**-1/2
max_eigval = sparse.linalg.eigsh(laplacian, k=1, return_eigenvectors=False)[0]
laplacian_ = 2.0 / max_eigval * laplacian - sparse.eye(N, dtype='float32') # linear interpolation [min_eigval, max_eigval] -> [-1, 1]

# Chebyshev polynomials of the first kind
support = [sparse.csr_matrix(sparse.eye(N, dtype='float32'))]
if (K > 0):
  support.append(laplacian_)
if (K > 1):
  for k in range(2, K + 1):
    support.append(2.0 * laplacian_.dot(support[k-1]) - support[k-2])

# Chebyshev coefficients for inverse wavelet
inverse_coeffs = []
for k in range(K + 1):
  func = lambda phi: cos(k * phi) * exp(-scale * max_eigval / 2.0 * (cos(phi) + 1.0))
  inverse_coeffs.append(2.0 / pi * integrate.quad(func, 0.0, pi, limit=500)[0])

# Chebyshev coefficients for wavelet
coeffs = []
for k in range(K + 1):
  func = lambda phi: cos(k * phi) * exp(scale * max_eigval / 2.0 * (cos(phi) + 1.0))
  coeffs.append(2.0 / pi * integrate.quad(func, 0.0, pi, limit=500)[0])

# inverse wavelet
inverse_wavelet = 0.5 * inverse_coeffs[0] * support[0]
for k in range(1, K + 1):
  inverse_wavelet += inverse_coeffs[k] * support[k]
inverse_wavelet.data[inverse_wavelet.data < thresold] = 0.0
indices = list(zip(inverse_wavelet.tocoo().row, inverse_wavelet.tocoo().col))
values = inverse_wavelet.data
dense_shape = inverse_wavelet.shape
inverse_wavelet = tf.SparseTensor(indices, values, dense_shape)

# wavelet
wavelet = 0.5 * coeffs[0] * support[0]
for k in range(1, K + 1):
  wavelet += coeffs[k] * support[k]
wavelet.data[wavelet.data < thresold] = 0.0
indices = list(zip(wavelet.tocoo().row, wavelet.tocoo().col))
values = wavelet.data
dense_shape = wavelet.shape
wavelet = tf.SparseTensor(indices, values, dense_shape)

print('Wavelets are computed!')

class ReLU_layer:

  def __call__(self, tensor, _):
    return tf.nn.relu(tensor)

class FC_layer:

  def __init__(self, indim, outdim):
    initial_value = tf.initializers.he_normal()((indim, outdim,))
    self.weight = tf.Variable(initial_value=initial_value, trainable=True)

  def __call__(self, tensor, _):
    return tf.linalg.matmul(tensor, self.weight)

class GC_layer:

  def __init__(self, N):
    initial_value = tf.ones((N, 1))
    self.weight = tf.Variable(initial_value=initial_value, trainable=True)

  def __call__(self, tensor, support):
    inverse_wavelet, wavelet = support
    tensor = tf.sparse.sparse_dense_matmul(inverse_wavelet, tensor)
    tensor = self.weight * tensor
    tensor = tf.sparse.sparse_dense_matmul(wavelet, tensor)
    return tensor

class Model:

  def __init__(self, optimizer, loss, accuracy):
    self.layers = []
    self.optimizer = optimizer
    self.loss = loss
    self.accuracy = accuracy
  
  def __call__(self, layer):
    self.layers.append(layer)

  def predict(self, signal, support):
    for layer in self.layers:
      signal = layer(signal, support)
    return signal

  def train(self, feature_matrix, support, train_matrix, train_mask, valid_matrix, valid_mask, epochs):

    sources = [layer.weight for layer in self.layers if type(layer) is FC_layer] \
            + [layer.weight for layer in self.layers if type(layer) is GC_layer]  # :-(
    
    for epoch in range(epochs):
      with tf.GradientTape() as tape:
        predictions = self.predict(feature_matrix, support)
        train_loss = self.loss(predictions, train_matrix, train_mask)
        train_accuracy = self.accuracy(predictions, train_matrix, train_mask)
        valid_loss = self.loss(predictions, valid_matrix, valid_mask)
        valid_accuracy = self.accuracy(predictions, valid_matrix, valid_mask)
      grads = tape.gradient(train_loss, sources)
      self.optimizer.apply_gradients(zip(grads, sources))
      print('%d epoch: train accuracy = %f%%, valid accuracy = %f%%' \
            % (epoch, 100 * train_accuracy.numpy(), 100 * valid_accuracy.numpy()))

  def test(self, feature_matrix, support, test_matrix, test_mask):
    predictions = self.predict(feature_matrix, support)
    test_accuracy = self.accuracy(predictions, test_matrix, test_mask)
    print('test accuracy = %f%%' % (100 * test_accuracy.numpy()))

# evaluation

def masked_loss(predictions, labels, mask):
  loss = tf.nn.softmax_cross_entropy_with_logits(labels, predictions)
  loss = tf.expand_dims(loss, 1)
  loss *= mask / tf.reduce_mean(mask)
  return tf.reduce_mean(loss)

def masked_accuracy(predictions, labels, mask):
  accuracy = tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1)), 'float32')
  accuracy = tf.expand_dims(accuracy, 1)
  accuracy *= mask / tf.reduce_mean(mask)
  return tf.reduce_mean(accuracy)

# builds the model
model = Model(tf.optimizers.Adam(learning_rate=learning_rate), masked_loss, masked_accuracy)
model(FC_layer(feature_matrix.shape[1], hidden))
model(GC_layer(feature_matrix.shape[0]))
model(ReLU_layer())
model(FC_layer(hidden, label_matrix.shape[1]))
model(GC_layer(feature_matrix.shape[0]))

# trains the model
model.train(feature_matrix, (inverse_wavelet, wavelet), train_matrix, train_mask, test_matrix, test_mask, epochs)

# tests the model
model.test(feature_matrix, (inverse_wavelet, wavelet), test_matrix, test_mask)
