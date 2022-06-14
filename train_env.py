import numpy as np
import tensorflow as tf
import tfDAT
import matplotlib.pyplot as plt
import os
import scipy.io

# get name of all mat files in the directory
all_env = os.listdir('data')
all_env.sort()

# Load the mat files
data = [scipy.io.loadmat(''.join(['data/',f])) for f in all_env if 'env' in f]
all_envelopes = [np.squeeze(f['env_ds']) for f in data]
all_envelopes = [np.diff(np.squeeze(np.array(env))) for env in all_envelopes]
min_len = len(min(all_envelopes,key=len))
all_envelopes = np.concatenate([env[np.newaxis,:min_len, np.newaxis] for env in all_envelopes], axis=0)
nenvs = len(all_envelopes)

all_idx = np.random.choice(nenvs, nenvs, replace=False)
train_idx = all_idx[:150]
val_idx = all_idx[150:]

# declare the model's stimulus values
fs = 100
dur = min_len/fs
stim_values = all_envelopes[train_idx]
stim_values = tf.convert_to_tensor(stim_values, dtype=tf.float32)
stim_values = tf.complex(stim_values, tf.zeros_like(stim_values))

val_values = all_envelopes[val_idx]
val_values = tf.convert_to_tensor(val_values, dtype=tf.float32)
val_values = tf.complex(val_values, tf.zeros_like(val_values))

# declare the model's target values
target_values = all_envelopes[train_idx,:-1]
target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)

# define the stimulus object
s = tfDAT.stimulus(values=stim_values, fs=fs)

# oscillator parameters and initial conditions
N = 1  # number of oscillators
osc_nat = 3  # natural frequency of oscillation in Hz
initconds = tf.constant(0.1 + 1j * 0.0, dtype=tf.complex64, shape=(N,))
l_params_dict = {
    "alpha": tf.Variable(
        0.1, dtype=tf.float32, constraint=lambda z: tf.clip_by_value(z, 0.0, np.inf)
    ),
    "beta1": tf.Variable(
        -1.0, dtype=tf.float32, constraint=lambda z: tf.clip_by_value(z, -np.inf, 0.0)
    ),
    "beta2": tf.constant(0.0, dtype=tf.float32),
    "delta": tf.constant(0.0, dtype=tf.float32),
    "cz": tf.Variable(1.0, dtype=tf.float32),
    "cw": tf.Variable(1.0, dtype=tf.float32),
    "cr": tf.Variable(
        1.0, dtype=tf.float32, constraint=lambda z: tf.clip_by_value(z, 0.0, np.inf)
    ),
    "w0": tf.Variable(2 * np.pi * osc_nat, dtype=tf.float32),
    "epsilon": tf.constant(1.0, dtype=tf.float32),
}
layer1 = tfDAT.neurons(
    params=l_params_dict, freqs=[l_params_dict["w0"] / (2 * np.pi)], initconds=initconds
)

# define the model
lr = 0.1
GrFNN = tfDAT.Model(
    optim=tf.optimizers.Adam(lr), layers=[layer1], stim=s, target=target_values
)

# let's integrate the model to see how it behaves before training
y_hat, freqs = GrFNN.inference(val_values, tf.float32)
plt.plot(GrFNN.time.numpy()[:-1], y_hat.numpy()[0], label='DAT')
plt.plot(GrFNN.time.numpy()[:-1], np.squeeze(target_values.numpy())[0], label='target')
plt.plot(GrFNN.time.numpy()[:-1], freqs.numpy()[0], label='DAT freq')
plt.legend()
plt.xlabel('time (s)')
plt.savefig('before_training.png')
plt.close()


# let's integrate and train
num_epochs = 10
for e in range(num_epochs):

    # 1. use GrFNN.inference to obtain y_hat

    # 2. use y_hat to calculate the trf_weights

    # 3. modify GrFNN.train_epoch to use the trf_weights

    print("==========================")
    print("==========================")
    print("Epoch: ", e + 1)
    print("--------------------------")
    print("alpha:    ", GrFNN.layers[0].params["alpha"].numpy())
    print("beta1:    ", GrFNN.layers[0].params["beta1"].numpy())
    print("beta2:    ", GrFNN.layers[0].params["beta2"].numpy())
    print("delta:    ", GrFNN.layers[0].params["delta"].numpy())
    print("cz:       ", GrFNN.layers[0].params["cz"].numpy())
    print("cw:       ", GrFNN.layers[0].params["cw"].numpy())
    print("cr:       ", GrFNN.layers[0].params["cr"].numpy())
    print("w0:       ", GrFNN.layers[0].params["w0"].numpy())
    print("w0/(2*pi):", GrFNN.layers[0].params["w0"].numpy() / (2 * np.pi))
    GrFNN.train_epoch(tf.float32)

# let's integrate the model to see how it behaves AFTER training
y_hat, freqs = GrFNN.inference(val_values, tf.float32)
plt.plot(GrFNN.time.numpy()[:-1], y_hat.numpy()[0], label='DAT')
plt.plot(GrFNN.time.numpy()[:-1], np.squeeze(target_values.numpy())[0], label='target')
plt.plot(GrFNN.time.numpy()[:-1], freqs.numpy()[0], label='DAT freq')
plt.legend()
plt.xlabel('time (s)')
plt.savefig('after_training.png')
plt.close()
