import tensorflow as tf
import numpy as np

###########################
# ODE function definition #
###########################
def dat_fun(
    t,
    x_y,
    alpha,
    beta1,
    beta2,
    delta,
    cz,
    cw,
    cr,
    w0,
    epsilon,
    sources_state,
    dtype=tf.float32,
):

    # keep some parameters always positive
    omega = tf.constant(2 * np.pi, dtype=dtype)

    x, y, freqs = tf.split(x_y, 3, axis=1)

    x2plusy2 = tf.add(tf.pow(x, 2), tf.pow(y, 2))
    x2plusy2squared = tf.pow(x2plusy2, 2)
    HOT = tf.divide(
        tf.multiply(tf.multiply(epsilon, beta2), x2plusy2squared),
        tf.add(tf.constant(1.0, dtype=dtype), -tf.multiply(epsilon, x2plusy2)),
    )

    xnew = tf.add_n(
        [
            tf.multiply(alpha, x),
            tf.multiply(omega, tf.multiply(-1.0, y)),
            tf.multiply(tf.multiply(delta, tf.multiply(-1.0, y)), x2plusy2),
            tf.multiply(beta1, tf.multiply(x, x2plusy2)),
            tf.multiply(x, HOT),
        ]
    )

    ynew = tf.add_n(
        [
            tf.multiply(alpha, y),
            tf.multiply(omega, x),
            tf.multiply(beta1, tf.multiply(y, x2plusy2)),
            tf.multiply(tf.multiply(delta, x), x2plusy2),
            tf.multiply(y, HOT),
        ]
    )

    # compute input
    sr, si = tf.split(sources_state[0], 2, axis=1)
    csr = tf.multiply(cz, sr)
    csi = tf.multiply(cz, si)
    csr = tf.multiply(
        csr,
        tf.add(
            tf.divide(1.0, tf.add(tf.pow(tf.add(1.0, -x), 2), tf.pow(y, 2))),
            -tf.divide(x, tf.add(tf.pow(tf.add(1.0, -x), 2), tf.pow(y, 2))),
        ),
    )
    csi = tf.multiply(
        csi,
        tf.multiply(
            -1.0, tf.divide(y, tf.add(tf.pow(tf.add(1.0, -x), 2), tf.pow(y, 2)))
        ),
    )

    xnew = tf.multiply(freqs, tf.add(xnew, csr))
    ynew = tf.multiply(freqs, tf.add(ynew, csi))
    xnew_ynew = tf.concat([xnew, ynew], axis=1)

    w = tf.multiply(freqs, 2 * np.pi)
    wnew = tf.add(
        -tf.divide(
            tf.multiply(
                cw,
                tf.multiply(
                    tf.sin(tf.math.angle(tf.complex(x, y))),
                    tf.multiply(
                        sr,
                        tf.add(
                            tf.divide(
                                1.0, tf.add(tf.pow(tf.add(1.0, -x), 2), tf.pow(y, 2))
                            ),
                            -tf.divide(
                                x, tf.add(tf.pow(tf.add(1.0, -x), 2), tf.pow(y, 2))
                            ),
                        ),
                    ),
                ),
            ),
            tf.abs(tf.complex(x, y)),
        ),
        -tf.multiply(cr, tf.divide((w - w0), w0)),
    )

    wnew = tf.multiply(freqs, wnew)
    dxdt_dydt = xnew_ynew
    dxdt_dydt_dwdt = tf.concat([dxdt_dydt, tf.divide(wnew, 2 * np.pi)], axis=1)

    return dxdt_dydt_dwdt
