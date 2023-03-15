"""
Impletation for 
"Enumeration and Identification of Active Users for Grant-Free NOMA Using Deep Neural Networks, figure 8"
run main file to train a network or load weights to skip training process
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from numba import jit, prange
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import HeNormal as H
import matplotlib.pyplot as plt
from matplotlib import ticker
from utils.paint import Plot
# policy = tf.keras.mixed_precision.Policy("float32")
# tf.keras.mixed_precision.set_global_policy(policy)


# %% functions
class FLAG():
    def __init__(self):
        self.snr = 10
        self.B = 1000
        self.fading = 'Rayleigh'
        self.Nd = 4
        self.K = 100
        self.Ka = 4
        self.m = 10
        self.dv = 2


def SensingMatrix(K, m, dv, code, seed):
    np.random.seed(seed)
    index = np.zeros((dv, K), dtype=int)
    Codebook = np.zeros((K, m)).astype('complex64')
    for i in range(K):
        index[:, i] = np.random.choice(m, dv, replace=False)
        for j in range(dv):
            Codebook[i, :][index[j, i]] = np.random.randn(1) + 1j*np.random.randn(1)
        # Codebook[i, :] = Codebook[i, :] / (np.sum(np.abs(Codebook[i, :])**2))**0.5
    return Codebook


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def gen_Rayleigh(K, m, Nd, B, p, Ka, snr):
    snrLinear = 10**(snr/10)    # 1 is signal power
    activeUser = np.zeros((B*p, K), dtype='int8')
    receiveSignal = np.zeros((B*p, Nd, 2*m), dtype='float32')
    symbol_set = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype='complex64') / np.sqrt(2)
    # symbol_set = np.array([1, -1j, -1, 1j], dtype='complex64') / np.sqrt(2)
    y_tilde = np.zeros((B*p, Nd, m), dtype='complex64')
    pilot_length = 0
    for i in prange(B*p):
        to_do = np.random.choice(K, Ka, replace=False)
        activeUser[i, :][to_do] = 1
        
        for jj in prange(Ka):
            bits = np.zeros((Nd, 1), dtype='complex64')
            bits[0:pilot_length, 0] = symbol_set[0]
            # l = (np.random.choice(Nd-pilot_length,1))[0]
            l = Nd - pilot_length
            symbol_index = np.random.choice(4, l)
            bits[pilot_length:l+pilot_length, 0] = symbol_set[symbol_index]
            channel = np.sqrt(1/2) * (np.random.randn(Nd, m) + 1j*np.random.randn(Nd, m))
            temp = bits * Codebook[to_do[jj], :, :] * channel
            y_tilde[i, :, :] += temp
    
    rxPower = np.sum(np.abs(y_tilde)**2) / y_tilde.size
    noisePower = rxPower / snrLinear
    noisePower = 1 / snrLinear
    y_tilde += np.sqrt(noisePower/2) * \
        (np.random.randn(B*p, Nd, m) + 1j * np.random.randn(B*p, Nd, m))
    receiveSignal[:, :, :m] = np.abs(y_tilde)
    receiveSignal[:, :, m:] = np.angle(y_tilde)
    return receiveSignal, activeUser


def network(args, alpha, flag):
    K, Nd, m = args.K, args.Nd, args.m
    def Hidden_Layer(inputs, alpha):
        x = layers.Dense(alpha, kernel_initializer=H())(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(alpha, kernel_initializer=H())(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, inputs])
        x = layers.ReLU()(x)
        return x

    model_input = layers.Input(shape=(Nd, 2*m), name='InputLayer')
    x1 = layers.Conv1D(64, 4, 4, kernel_initializer=H())(model_input)
    x1 = layers.ReLU()(x1)
    x1 = layers.Flatten()(x1)
    x2 = layers.Conv1D(64, 2, 2, kernel_initializer=H())(model_input)
    x2 = layers.ReLU()(x2)
    x2 = layers.Flatten()(x2)
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(alpha, kernel_initializer=H())(x)
    x = layers.BatchNormalization()(x)
    x = Hidden_Layer(x, alpha)
    x = Hidden_Layer(x, alpha)

    x = layers.Dense(K, kernel_initializer=H())(x)
    if flag == 'AUE':
        x = layers.Softmax()(x)
    else:
        x = layers.Activation('sigmoid', name='sig')(x)
    
    return tf.keras.Model(model_input, x)


# %% params and Data
args = FLAG()
args.fading = 'Rayleigh'
args.Ka = 4
args.Nd = 8
args.m = 10
args.dv = 2
args.snr = 10
Codebook = np.zeros((args.K, args.Nd, args.m), 'complex64')
for i in range(args.Nd):
    Codebook[:, i, :] = SensingMatrix(args.K, args.m, args.dv, 'LDS', i)
Codebook = Codebook.astype('complex64')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=5*10**-7)
early_stop = EarlyStopping(monitor='val_loss', patience=15)
# K, m, Nd, B, p, Ka, snr = args.K, args.m, args.Nd, args.B, 1, args.Ka, args.snr
# fading, stage = 'Rayleigh', 'train'

t = time.time()
p = 9000
Y, Alpha = gen_Rayleigh(args.K, args.m, args.Nd, args.B, p, args.Ka, args.snr)
    
# np.random.seed(100)
# np.random.shuffle(Y)
# np.random.seed(100)
# np.random.shuffle(Alpha)
print('time elapsed: ' + str(time.time() - t))


# %% build model
from keras_flops import get_flops
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False)
net = network(args, 1000, 'AUI')
_ = tf.keras.utils.plot_model(net, to_file='./record/structure.png', show_shapes=1)
flops = get_flops(net, batch_size=1)
print('FLOPS: ' + str(flops/1e6))


# %% train model
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)
t = time.time()
net.compile(optimizer=optimizers.Adam(1e-3),
            loss=tf.losses.BinaryCrossentropy(label_smoothing=0.0),
            metrics='acc')
net.fit(Y, Alpha,
        epochs=20,
        batch_size=args.B,
        validation_split=0.05,
        callbacks=[reduce_lr, early_stop],verbose=1)
print('Elaspsed time: ' + str((time.time() - t)/60) + ' min')
# net.save_weights('AUI_k' + str(args.Ka) + '_Nd' + str(args.Nd) + '4.h5')
# net.load_weights('AUI_k' + str(args.Ka) + '_Nd' + str(args.Nd) + '4.h5')

hist_dict = net.history.history
history = dict(loss=[], val_loss=[], acc=[], val_acc=[])
history = {key: history.get(key, []) + hist_dict.get(key, []) for key in set(list(history.keys()) + list(hist_dict.keys()))}
history['AER'] = history.pop('acc')
history['val_AER'] = history.pop('val_acc')
Plot.train_history(history)


# %% visualize
test_SNR = np.arange(0, 21, 2)
p = 100
recall = np.zeros(test_SNR.shape[0])
falarm = np.zeros(test_SNR.shape[0])
aer = np.zeros(test_SNR.shape[0])
for i in range(len(test_SNR)):
    print('|', test_SNR[i], end='') if i != (len(test_SNR)-1) else print('|', test_SNR[i])
    a, b = gen_Rayleigh(args.K, args.m, args.Nd, args.B, p, args.Ka, test_SNR[i])
    # a, b = Train_Loo(N, m, Nd, 1024, p, k, test_SNR[i], 'test')
    p_pred = net.predict(a, batch_size=args.B)
    
    p_hat = np.argsort(-p_pred)[:, :args.Ka]
    temp = np.zeros([p*args.B, args.K], dtype='int')
    for j in range(args.Ka):
        temp[np.arange(p*args.B), p_hat[:, j]] = 1
    aer[i] = np.sum(np.abs(temp - b)) / b.size
    temp[temp==0] = -1
    TP = np.where((b - temp)==0)[0].size
    FN = np.where((b - temp)==2)[0].size
    TN = np.where((b - temp)==1)[0].size
    FP = np.where((b - temp)==-1)[0].size
    recall[i] = TP / (TP+FN)
    falarm[i] = FP / (FP+TN)
    


fig8 = np.array([0.499, 0.643, 0.766, 0.867, 0.937, 0.970, 0.988, 0.995, 0.995, 0.997, 0.993])
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=10)          # controls default text sizes
plt.rc('font', size=12)          # controls default text sizes
fig, ax = plt.subplots()
fig.set_size_inches(4.5, 3)
kw = dict(markersize=9, markerfacecolor='none', markeredgewidth=1, linewidth=.8, clip_on=False)
ax.plot(test_SNR, recall, 'b-o', label='simulation', **kw)
ax.plot(test_SNR, fig8, 'k->', label='paper', **kw)
ax.grid('true', linestyle = ':', alpha=.5)
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('recall')
ax.set_xlim(min(test_SNR), max(test_SNR))
ax.set_ylim(0, 1)
ax.set_xticks(np.arange(0, 21, 5))
ax.legend(loc='lower right')
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.tick_params(axis="both", which='both', direction="in", pad=3)
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%.1f'))
plt.show()
filename = 'AUE' + 'Ka_' + str(args.Ka) + 'Nd_' + str(args.Nd) + '.txt'
# np.savetxt(filename, np.expand_dims(aer, 1).T, '%10.8f',delimiter=',')


 