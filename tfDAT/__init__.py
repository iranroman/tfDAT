import tensorflow as tf
import numpy as np
from .utils import *
from .ode_funcs import dat_fun


########################################
# object to define the stimulus object #
########################################
class stimulus:
    def __init__(
        self,
        values=None,
        fs=None,
    ):

        self.values = values
        vshape = tf.shape(self.values)
        self.ndatapoints = vshape[0]
        self.nsamps = tf.cast(vshape[1], dtype=tf.float32)
        self.fs = tf.constant(fs, dtype=tf.float32)
        self.dt = tf.constant(1.0 / self.fs, dtype=tf.float32)
        self.dur = tf.constant(self.nsamps / self.fs, dtype=tf.float32)


###############################
# layer of oscillators object #
###############################
class neurons:
    def __init__(
        self,
        params=None,
        freqs=None,
        initconds=tf.constant(0, dtype=tf.complex64, shape=(256,)),
    ):

        self.params = params
        self.initconds = initconds
        self.params["freqs"] = freqs
        self.N = 1


####################################
# Object to define the GrFNN model #
####################################
class Model:
    def __init__(
        self,
        layers=None,
        stim=None,
        zfun=dat_fun,
        optim=None,
        target=None,
    ):

        self.optim = optim
        self.target = target
        self.layers = layers
        self.stim = stim
        self.zfun = zfun
        self.dt = self.stim.dt
        self.half_dt = self.dt / 2
        self.nsamps = self.stim.nsamps
        self.dur = self.stim.dur
        self.time = tf.range(self.dur, delta=self.dt, dtype=tf.float32)

    @tf.function()
    def train_epoch(self, dtype=tf.float16):

        time = tf.cast(self.time, dtype)
        stim_values = tf.cast(complex2concat(self.stim.values, 2), dtype)
        layer_state = [
            tf.tile(
                tf.expand_dims(
                    tf.cast(
                        tf.concat(
                            [
                                complex2concat(layer.initconds, 0),
                                [layer.params["w0"] / (2 * np.pi)],
                            ],
                            axis=0,
                        ),
                        dtype,
                    ),
                    axis=0,
                ),
                tf.constant([self.stim.ndatapoints.numpy(), 1]),
            )
            for layer in self.layers
        ]
        layer_alpha = [tf.cast(layer.params["alpha"], dtype) for layer in self.layers][
            0
        ]
        layer_beta1 = [tf.cast(layer.params["beta1"], dtype) for layer in self.layers][
            0
        ]
        layer_beta2 = [tf.cast(layer.params["beta2"], dtype) for layer in self.layers][
            0
        ]
        layer_delta = [tf.cast(layer.params["delta"], dtype) for layer in self.layers][
            0
        ]
        layer_cz = [tf.cast(layer.params["cz"], dtype) for layer in self.layers][0]
        layer_cw = [tf.cast(layer.params["cw"], dtype) for layer in self.layers][0]
        layer_cr = [tf.cast(layer.params["cr"], dtype) for layer in self.layers][0]
        layer_w0 = [tf.cast(layer.params["w0"], dtype) for layer in self.layers][0]
        layer_epsilon = [
            tf.cast(layer.params["epsilon"], dtype) for layer in self.layers
        ][0]
        zfun = self.zfun
        optim = self.optim
        target = self.target

        train_step(
            optim,
            target,
            time,
            layer_state,
            layer_alpha,
            layer_beta1,
            layer_beta2,
            layer_delta,
            layer_cz,
            layer_cw,
            layer_cr,
            layer_w0,
            layer_epsilon,
            zfun,
            stim_values,
            dtype,
        )

    def inference(self, stim_values, dtype=tf.float16):

        time = tf.cast(self.time, dtype)
        stim_values = tf.cast(complex2concat(stim_values, 2), dtype)
        layer_state = [
            tf.tile(
                tf.expand_dims(
                    tf.cast(
                        tf.concat(
                            [
                                complex2concat(layer.initconds, 0),
                                [layer.params["w0"] / (2 * np.pi)],
                            ],
                            axis=0,
                        ),
                        dtype,
                    ),
                    axis=0,
                ),
                tf.constant([tf.shape(stim_values).numpy()[0], 1]),
            )
            for layer in self.layers
        ]
        layer_alpha = [tf.cast(layer.params["alpha"], dtype) for layer in self.layers][
            0
        ]
        layer_beta1 = [tf.cast(layer.params["beta1"], dtype) for layer in self.layers][
            0
        ]
        layer_beta2 = [tf.cast(layer.params["beta2"], dtype) for layer in self.layers][
            0
        ]
        layer_delta = [tf.cast(layer.params["delta"], dtype) for layer in self.layers][
            0
        ]
        layer_cz = [tf.cast(layer.params["cz"], dtype) for layer in self.layers][0]
        layer_cw = [tf.cast(layer.params["cw"], dtype) for layer in self.layers][0]
        layer_cr = [tf.cast(layer.params["cr"], dtype) for layer in self.layers][0]
        layer_w0 = [tf.cast(layer.params["w0"], dtype) for layer in self.layers][0]
        layer_epsilon = [
            tf.cast(layer.params["epsilon"], dtype) for layer in self.layers
        ][0]
        zfun = self.zfun

        y_hat, freqs = forward(
            time,
            layer_state,
            layer_alpha,
            layer_beta1,
            layer_beta2,
            layer_delta,
            layer_cz,
            layer_cw,
            layer_cr,
            layer_w0,
            layer_epsilon,
            zfun,
            stim_values,
            dtype,
        )
        return tf.squeeze(y_hat), tf.squeeze(freqs)
