"""
Impletation for Deep Neural Network-Based Active User Detection for Grant-Free NOMA Systems
"""
from numba import jit, prange
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.paint import Plot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import os
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# %%
def AUD(alpha, N, m, Nd):
    def Hidden_Layer(inputTensor, alpha, stage):
        name_base = 'HL' + stage
        x = layers.Dense(alpha, name=name_base + '_1')(inputTensor)
        x = layers.BatchNormalization(name=name_base + '_2')(x)
        x = layers.Activation('relu', name=name_base + '_3')(x)
        x = layers.Dropout(0.1, name=name_base + '_4')(x)
        x = layers.add([x, inputTensor], name=name_base + '_Add')
        return x

    model_input = layers.Input(shape=(Nd*2*m, ), name='InputLayer')
    x = layers.Dense(alpha, name='InputFC')(model_input)
    x = layers.BatchNormalization(name='InputBN')(x)
    x = Hidden_Layer(x, alpha, stage='A')
    x = Hidden_Layer(x, alpha, stage='B')
    x = Hidden_Layer(x, alpha, stage='C')
    x = Hidden_Layer(x, alpha, stage='D')
    x = Hidden_Layer(x, alpha, stage='E')
    x = Hidden_Layer(x, alpha, stage='F')

    x = layers.Dense(N, name='OutputFC')(x)
    x = layers.Softmax(name='OutputActivation')(x)
    model = Model(model_input, x, name='D_AUD')
    return model


def CodebookGen(K, m, dv, code, seed):
    np.random.seed(seed)
    tx_pwr = 23    # dbm
    tx_pwr = 10**((tx_pwr-30)/10)
    if code == 'LDS':
        index = np.zeros((dv, K), dtype=int)
        Codebook = np.zeros((K, m)).astype('complex128')
        for i in range(K):
            index[:, i] = np.random.choice(m, dv, replace=False)
            for j in range(dv):
                Codebook[i, :][index[j, i]] = np.random.randn(1) + 1j*np.random.randn(1)
            Codebook[i, :] = Codebook[i, :] / (np.sum(np.abs(Codebook[i, :])**2))**0.5
    elif code == 'CN':
        Codebook = np.random.randn(K, m) + 1j*np.random.randn(K, m)
        for i in range(K):
            Codebook[i, :] = Codebook[i, :] / (np.sum(np.abs(Codebook[i, :])**2))**0.5*(tx_pwr*dv)
    elif code == 'ZC':
        root = np.array([3, 9, 11, 13, 17, 19, 23, 27, 29, 31, 33, 37, 39, 41, 43, 47, 51, 53, 57, 59, 61, 67, 69])
        index = np.arange(0, m)
        Codebook = np.zeros((1, m), 'complex128')
        for i in range(root.size):
            seq = np.exp( -1j * np.pi * root[i] * index.T*(index+1) / m )
            for j in range(m):
                Codebook = np.concatenate((Codebook, np.expand_dims(np.roll(seq, j), 0)), axis=0)
        Codebook = Codebook[1:, :]
    return Codebook


@jit(nopython=True, parallel=True, fastmath=True, nogil=True, cache=True)
def gen_Rayleigh(K, m, Nd, B, p, Ka, snr):
    snrLinear = 10**(snr/10)    # 1 is signal power
    activeUser = np.zeros((B*p, K), dtype='int8')
    receiveSignal = np.zeros((B*p, Nd*2*m), dtype='float32')
    symbol_set = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype='complex64') / np.sqrt(2)
    y_tilde = np.zeros((B*p, Nd*m), dtype='complex64')
    pilot_length = 1
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
            temp = bits * Codebook[to_do[jj], :] * channel
            y_tilde[i, :] += temp.reshape(-1)
    
    rxPower = np.sum(np.abs(y_tilde)**2) / y_tilde.size
    noisePower = rxPower / snrLinear
    y_tilde += np.sqrt(noisePower/2) * \
        (np.random.randn(B*p, Nd*m) + 1j * np.random.randn(B*p, Nd*m))
    receiveSignal[:, :Nd*m] = np.real(y_tilde)
    receiveSignal[:, Nd*m:] = np.imag(y_tilde)
    return receiveSignal, activeUser


# %%
# Params
m = 70          # num of subcarriers
N = 100         # num of total users
Nd = 7          # num of "multiple measurement"
alpha = 10*N    # num of neurons each dense layer
p = 100     # num of data generated at one time (depend on memory)
tap = 1
k = 5
snr = 10
dv = 10         # num of spreading code for each user
Codebook = np.zeros((N, Nd, m), 'complex64')
for i in range(Nd):
    Codebook[:, i, :] = CodebookGen(N, m, dv, 'LDS', i)
Codebook = Codebook.astype('complex64')
# Training data generation
print('codebook power: ' + str(np.sum(np.abs(Codebook[1, :])**2)))
receiveSignal, activeUser = gen_Rayleigh(N, m, Nd, 2048, 1000, k, snr)
activeUser = activeUser / k


# %%
# Training
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
saveWeight = ModelCheckpoint(filepath='./' + 'AUD_' + str(k) + 'user_' + str(dv) + 'dv_' + str(snr) + 'snr_' + '.h5',
                             monitor='loss',
                             # verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto', save_freq=150)
reduce_lr = ReduceLROnPlateau(monitor='loss',
                              factor=0.5, patience=3,
                              min_lr=0.5*10**-7)
adam = Adam(learning_rate=5*10**-4)


AUD1 = AUD(alpha, N, m, Nd)
AUD1.summary()
# model = tf.keras.utils.plot_model(AUD1)
AUD1.compile(optimizer=adam,
             loss='categorical_crossentropy',
             metrics='acc')

t = time.time()
AUD1.fit(receiveSignal, activeUser,
          epochs=500,
          batch_size=2048,
          validation_split=0.2,
          callbacks=[reduce_lr, early_stopping])
print('time elpased: ' + str((time.time() - t)/60))
# AUD1.save_weights('DNN_k' + str(k) + 'T' + str(Nd) + '.h5')
# AUD1.load_weights('DNN_k' + str(k) + 'T' + str(Nd) + '.h5')


# %%
hist_dict = AUD1.history.history
history = dict(loss=[], val_loss=[], acc=[], val_acc=[])
history = {key: history.get(key, []) + hist_dict.get(key, []) for key in set(list(history.keys()) + list(hist_dict.keys()))}
history['AER'] = history.pop('acc')
history['val_AER'] = history.pop('val_acc')
Plot.train_history(history)


# %%
test_SNR = np.arange(0, 21, 2)
p = 5
p_succ = np.zeros(test_SNR.shape[0])
aer = np.zeros(test_SNR.shape[0])
recall = np.zeros(test_SNR.shape[0])
aer = np.zeros(test_SNR.shape[0])
for i in range(len(test_SNR)):
    print('|', test_SNR[i], end='') if i != (len(test_SNR)-1) else print('|', test_SNR[i])
    a, b = gen_Rayleigh(N, m, Nd, 2048, p, k, test_SNR[i])
    # a, b = Train_Loo(N, m, Nd, 1024, p, k, test_SNR[i], 'test')
    p_pred = AUD1.predict(a, batch_size=2048)

    p_hat = np.argsort(-p_pred)[:, :k]
    temp = np.zeros([p*2048, N], dtype='int')
    for j in range(k):
        temp[np.arange(p*2048), p_hat[:, j]] = 1
    aer[i] = np.sum(np.abs(temp - b)) / b.size
    error = np.where(b != 0)[1].reshape(-1, k) - np.sort(p_hat)
    error[error!=0] = 1
    p_succ[i] = 1 - np.mean(error)
    temp[temp==0] = -1
    TP = np.where((b - temp)==0)[0].size
    FN = np.where((b - temp)==2)[0].size
    TN = np.where((b - temp)==1)[0].size
    FP = np.where((b - temp)==-1)[0].size
    recall[i] = TP / (TP+FN)

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=10)          # controls default text sizes
plt.rc('font', size=12)          # controls default text sizes
fig, ax = plt.subplots()

fig.set_size_inches(4.5, 3)
kw = dict(markersize=9, markerfacecolor='none', markeredgewidth=1, linewidth=.8, clip_on=False)
ax.plot(test_SNR, recall, 'b-o', label='simulation', **kw)
ax.grid('true', linestyle = ':', alpha=.5)
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('recall')
ax.set_xlim(min(test_SNR), max(test_SNR))
ax.set_ylim(0, 1)
ax.set_xticks(np.arange(0, 21, 5))
ax.legend(loc='lower right')
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.tick_params(axis="both", which='both', direction="in", pad=3)
plt.show()
# np.savetxt(filename, np.expand_dims(aer, 1).T, '%10.8f',delimiter=',')
# p_succ = np.loadtxt(filename,delimiter=',')

     
    
    
    
