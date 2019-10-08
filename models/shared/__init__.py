import tensorflow as tf

def make_pipe(pref, fst_slab, nth_slab, dropout, config, reducer, m, h, fcname = "FC", fczname = "FCZ", default_reuse = None):
    main = tf.layers.dropout(
        tf.nn.relu(
            tf.layers.dense(
                fst_slab(),
                h,
                name = (fcname + pref),
                reuse = default_reuse,
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
            )),
            dropout
    )

    outns = list()
    reuse=default_reuse
    for n in range(m):
        print(nth_slab(n).shape)
        outns.append(
            tf.layers.dropout(
                tf.nn.relu(
                    tf.layers.dense(
                        nth_slab(n),
                        h,
                        name = (fczname + pref),
                        reuse = reuse,
                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                    )),
                dropout
            )
        )
        reuse=True
    pool = reducer(outns, pref)
    #print(pool.shape)
    return (main, pool)
