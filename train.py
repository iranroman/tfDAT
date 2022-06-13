import numpy as np
import tensorflow as tf
import tfDAT
import matplotlib.pyplot as plt

# declare the model's stimulus values
fs = 100
dur = 5
t = np.linspace(0, dur, dur * fs, endpoint=False)[np.newaxis,:,np.newaxis] # stimulus must have 3 dimensions: [ndatapoints, time, nchannels]
stim_values = np.sin(2 * np.pi * t)
stim_values = tf.convert_to_tensor(stim_values, dtype=tf.float32)
stim_values = tf.complex(stim_values, tf.zeros_like(stim_values))

# declare the model's target values
target_values = np.sin(2 * np.pi * t[:, :-1])
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
y_hat, freqs = GrFNN.forward(tf.float32)
plt.plot(GrFNN.time.numpy()[:-1], y_hat.numpy(), label='DAT')
plt.plot(GrFNN.time.numpy()[:-1], np.squeeze(target_values.numpy()), label='target')
plt.plot(GrFNN.time.numpy()[:-1], freqs.numpy(), label='DAT freq')
plt.legend()
plt.xlabel('time (s)')
plt.savefig('before_training.png')
plt.close()


# let's integrate and train
num_epochs = 10
for e in range(num_epochs):

    # 1. use GrFNN.forward to obtain y_hat

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
y_hat, freqs = GrFNN.forward(tf.float32)
plt.plot(GrFNN.time.numpy()[:-1], y_hat.numpy(), label='DAT')
plt.plot(GrFNN.time.numpy()[:-1], np.squeeze(target_values.numpy()), label='target')
plt.plot(GrFNN.time.numpy()[:-1], freqs.numpy(), label='DAT freq')
plt.legend()
plt.xlabel('time (s)')
plt.savefig('after_training.png')
plt.close()
