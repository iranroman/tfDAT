import tensorflow as tf


def complex2concat(x, axis):
    return tf.concat([tf.math.real(x), tf.math.imag(x)], axis=axis)


def Runge_Kutta_4(
    time,
    layers_state,
    layers_alpha,
    layers_beta1,
    layers_beta2,
    layers_delta,
    layers_cz,
    layers_cw,
    layers_cr,
    layers_w0,
    layers_epsilon,
    zfun,
    stim_values,
    dtype=tf.float16,
):
    def scan_fn(layers_state, time_dts_stim):
        def get_next_k(time_val, layers_state):

            layers_k = [
                zfun(
                    time_val,
                    layer_state,
                    layers_alpha,
                    layers_beta1,
                    layers_beta2,
                    layers_delta,
                    layers_cz,
                    layers_cw,
                    layers_cr,
                    layers_w0,
                    layers_epsilon,
                    layers_state,
                    dtype,
                )
                for layer_state in layers_state[1:]
            ]

            return layers_k

        def update_states(time_scaling, layers_k0, layers_k, new_stim):

            layers_state = [
                tf.add(layer_k0, tf.scalar_mul(time_scaling, layer_k))
                for (layer_k0, layer_k) in zip(layers_k0, layers_k)
            ]
            layers_state.insert(0, new_stim)

            return layers_state

        t, dt, stim, stim_shift = time_dts_stim

        t_plus_half_dt = tf.add(t, dt / 2)
        t_plus_dt = tf.add(t, dt)

        layers_k0 = layers_state.copy()
        layers_state.insert(0, stim)

        layers_k1 = get_next_k(t, layers_state)
        layers_state = update_states(
            dt / 2, layers_k0, layers_k1, tf.divide(tf.add(stim, stim_shift), 2)
        )
        layers_k2 = get_next_k(t_plus_half_dt, layers_state)
        layers_state = update_states(
            dt / 2, layers_k0, layers_k2, tf.divide(tf.add(stim, stim_shift), 2)
        )
        layers_k3 = get_next_k(t_plus_half_dt, layers_state)
        layers_state = update_states(dt, layers_k0, layers_k3, stim_shift)
        layers_k4 = get_next_k(t_plus_dt, layers_state)

        layers_state = [
            tf.add(
                layer_k0,
                tf.multiply(
                    dt / 6,
                    tf.add_n(
                        [
                            layer_k1,
                            tf.scalar_mul(2, layer_k2),
                            tf.scalar_mul(2, layer_k3),
                            layer_k4,
                        ]
                    ),
                ),
            )
            for (layer_k0, layer_k1, layer_k2, layer_k3, layer_k4) in zip(
                layers_k0, layers_k1, layers_k2, layers_k3, layers_k4
            )
        ]
        # tf.print('=============')
        # tf.print(layers_state)

        return layers_state

    dts = time[1:] - time[:-1]
    layers_states = tf.scan(
        scan_fn,
        [
            time[:-1],
            dts,
            tf.transpose(stim_values[:, :-1, :], (1, 0, 2)),
            tf.transpose(stim_values[:, 1:, :], (1, 0, 2)),
        ],
        layers_state,
    )

    return layers_states


def train_step(
    optim,
    target,
    time,
    layers_state,
    layers_alpha,
    layers_beta1,
    layers_beta2,
    layers_delta,
    layers_cz,
    layers_cw,
    layers_cr,
    layers_w0,
    layers_epsilon,
    zfun,
    stim_values,
    dtype,
):
    with tf.GradientTape() as tape:
        mse = tf.losses.MeanSquaredError()
        # keep some parameters always positive
        layers_states = Runge_Kutta_4(
            time,
            layers_state,
            layers_alpha,
            layers_beta1,
            layers_beta2,
            layers_delta,
            layers_cz,
            layers_cw,
            layers_cr,
            layers_w0,
            layers_epsilon,
            zfun,
            stim_values,
            dtype,
        )

        l_output_r, l_output_i, freqs = tf.split(layers_states[0], 3, axis=2)
        l_output_r = tf.transpose(l_output_r, (1, 2, 0))
        l_output_i = tf.transpose(l_output_i, (1, 2, 0))
        freqs = tf.transpose(freqs, (1, 2, 0))
        y_hat = tf.squeeze(l_output_r, axis=1)
        
        # Eshed and Iran TODO:
        # 1. compute the boosting TRF using 'y_hat' as input and EEG as output
        # 2. use the TRF weights to convolve with 'y_hat'
        # 3. change the loss to be between the eeg and the output of the convolution 'y_hat'*TRF weights 
        # to achieve this in the tf.function scope we might have to do something like:
        # https://stackoverflow.com/questions/55679540/tensorflow-2-0-function-with-tf-function-decorator-doesnt-take-numpy-function
        # or write the boosting algorithm in tensorflow
        
        curr_loss = mse(target, y_hat)
    tf.print("==========================")
    tf.print("    Loss: ", curr_loss)
    tf.print("==========================")
    var_list = {
        "alpha ": layers_alpha,
        "beta1 ": layers_beta1,
        #'beta2 ': layers_beta2,
        #'delta ': layers_delta,
        "cz ": layers_cz,
        "cw ": layers_cw,
        "cr ": layers_cr,
        "w0 ": layers_w0,
    }
    grads = tape.gradient(curr_loss, list(var_list.values()))
    optim.apply_gradients(zip(grads, list(var_list.values())))
    return layers_states, tf.squeeze(l_output_r, axis=1), freqs, curr_loss, var_list


def forward(
    time,
    layers_state,
    layers_alpha,
    layers_beta1,
    layers_beta2,
    layers_delta,
    layers_cz,
    layers_cw,
    layers_cr,
    layers_w0,
    layers_epsilon,
    zfun,
    stim_values,
    dtype,
):

    layers_states = Runge_Kutta_4(
        time,
        layers_state,
        layers_alpha,
        layers_beta1,
        layers_beta2,
        layers_delta,
        layers_cz,
        layers_cw,
        layers_cr,
        layers_w0,
        layers_epsilon,
        zfun,
        stim_values,
        dtype,
    )

    l_output_r, l_output_i, freqs = tf.split(layers_states[0], 3, axis=2)
    l_output_r = tf.transpose(l_output_r, (1, 2, 0))
    l_output_i = tf.transpose(l_output_i, (1, 2, 0))
    y_hat = tf.squeeze(l_output_r, axis=1)

    return y_hat, freqs
