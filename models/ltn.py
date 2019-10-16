import tensorflow as tf
from nus_johnson.types import *
from models.shared import *

class ltn(object):
    def __init__(self, input_types, input_shapes, m, tau, embeddings_width, config):
        v = 0
        nv = 0
        wv = 0
        wnv = 0
        self.testing = tf.placeholder(tf.bool)
        if (config["arch"] in ["ltn+vecs", "ltn+allvecs"]):
            if (embeddings_width > 0):
                wv = 1
            else:
                v = 1
                
        if (config["arch"] in ["ltn+allvecs"]):
            if (embeddings_width > 0):
                wnv = 1
            else:
                nv = 1

        self._parameters = config
        self.handle = tf.placeholder(tf.string, shape=[])
        handle_iterator = tf.data.Iterator.from_string_handle(self.handle,
                                                              ltn_gen_type(m, v, nv, wv, wnv),
                                                              ltn_gen_shape(embeddings_width)(m, v, nv, wv, wnv, tau))

        data = handle_iterator.get_next()
        self.y = tf.to_float(ltn_gen_get_truth(m, v, wv)(data), name='ToFloat')
        self.dropout = tf.placeholder(tf.float32)
        num_classes = truth_shape[1]

        def augment(a,b):
            return tf.concat([a,
                              tf.reshape(tf.to_float(b, name='ToFloat'), [-1, tau])
            ],axis =1)

        def fst_slab():
            res = ltn_gen_get_main_feat(m, v, wv)(data)

            if (v == 1):
                res = augment(res,
                               ltn_gen_get_main_vec(m, v, wv)(data))
            if (wv == 1):
                res = augment(res,
                               tf.cast(ltn_gen_get_main_w2vec(m, v, wv)(data), tf.float32))
            return res

        def nth_slab(n):
            res = ltn_gen_get_neigh_feats(m, v, wv)(data)[n]
            if (v == 1):
                res = augment(res,
                               ltn_gen_get_neigh_vecs(m, v, wv)(data)[n])
            if (wv == 1):
                res = augment(res,
                               tf.cast(ltn_gen_get_neigh_w2vecs(m, v, wv)(data)[n], tf.float32))
            return res

        h = config["h"]
        if (config["arch"] == "rtn"):
            def reducer(outns, scope):
                inputs = list(outns)
                cell = tf.nn.rnn_cell.LSTMCell(config["h"], activation="linear")
                out, state = tf.nn.static_rnn(cell=cell, inputs=inputs, dtype=tf.float32, scope="rnn"+scope)
                return out[-1]
        else:
            if (config["arch"] in ["ltn", "ltn+allvecs", "ltn+vecs"]):
                reducer = lambda outns, scope: tf.reduce_max(outns, axis = 0)
            else:
                print("Unknown arch", config["arch"])
                sys.exit()

        self.main, self.pool = make_pipe("", fst_slab, nth_slab, self.dropout, None, reducer, m, h)


        self.conc = tf.concat([self.main, self.pool], axis=1)
        self.model = self.fully = tf.layers.dense(self.conc,
                                                      num_classes, name="OUT", reuse=None,
                                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in (["FCZ", "FCZ'", "FC", "OUT", "rnn", "OUTRNN/lstm_cell"])]

        kernels_by_shape = [["FCZ/kernel:0", "FC/kernel:0"],
                            ["OUT/kernel:0"],
                            ["FC'/kernel:0"],
                            ["OUTRNN/lstm_cell/kernel:0"],
                            ["rnn/lstm_cell/kernel:0"]]

        kernel_names_to_vars = lambda kernel_names: [v for v in tf.trainable_variables()
                                                     if v.name in kernel_names]
        zomg = lambda l2reg, kernels: l2reg*tf.nn.l2_loss(kernel_names_to_vars(kernels))

        l2_coeffs = map(lambda x: zomg(config["l2_reg"], x),
                        kernels_by_shape)

        l2_term = sum(l2_coeffs)

        self.rawout = self.model
        self.scaled = tf.nn.sigmoid(self.rawout)

        # Op for calculating the loss
        with tf.name_scope("cross_ent"):
            self.loss_op = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,
                                                        logits=self.rawout)
                + l2_term)


        def make_train_op(var_list):
            # Get gradients of all trainable variables
            gradients = tf.gradients(self.loss_op, var_list)
            gradients = list(zip(gradients, var_list))
            optimizer = tf.train.RMSPropOptimizer(learning_rate=config["learning_rate"],
                                                  decay=config["decay_rate"])
            if (config["arch"] == "rtn"):
                optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0)
            return optimizer.apply_gradients(grads_and_vars=gradients)

        # Train op
        with tf.name_scope("train"):
            self.train_op = make_train_op(var_list)
            self.second_pass_train_op = make_train_op(
                [v for v in tf.trainable_variables() if v.name.split('/')[0] in (["OUT", "rnn", "OUTRNN/lstm_cell"])]
            )

    def train_hook(self, sess, handle):
        return sess.run([self.train_op, self.loss_op],
                        feed_dict={self.handle: handle,
                                   self.dropout: self._parameters["dropout_rate"],
                                   self.testing: False})[1]
    def second_pass_train_hook(self, sess, handle):
        return sess.run([self.second_pass_train_op, self.loss_op],
                        feed_dict={self.handle: handle,
                                   self.dropout: self._parameters["dropout_rate"],
                                   self.testing: False})[1]
    def eval_psgl_hook(self, sess, handle):
        return sess.run([self.rawout, self.scaled, self.y,
                         self.loss_op], feed_dict={self.handle: handle,
                                                   self.dropout: 1.,
                                                   self.testing: True})
