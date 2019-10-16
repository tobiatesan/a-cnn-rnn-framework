import tensorflow as tf
from nus_johnson.types import *
from models.shared import *
import sys

class lbraid(object):
    def __init__(self, input_types, input_shapes, m, tau, embeddings_width, config):
        self._parameters = config
        self.handle = tf.placeholder(tf.string, shape=[])
        tau = config["tau"]
        h = config["h"]
        wv = (embeddings_width > 0)
        wnv = (embeddings_width > 0)
        v = 1 - wv
        nv = 1 - wnv

        handle_iterator = tf.data.Iterator.from_string_handle(self.handle,
                                                              ltn_gen_type(m, v, nv, wv, wnv),
                                                              ltn_gen_shape(embeddings_width)(m, v, nv, wv, wnv, tau))
        LEN = tau
        if (embeddings_width > 0):
            LEN = embeddings_width


        data = handle_iterator.get_next()
        self.y = tf.to_float(ltn_gen_get_truth(m, v, wv)(data), name='ToFloat')
        self.dropout = tf.placeholder(tf.float32)
        num_classes = truth_shape[1]

        
        def pack_vec(vec):
            return tf.reshape(tf.to_float(vec, name='ToFloat'), [-1, LEN])

        def fst_feat():
            return ltn_gen_get_main_feat(m, v, wv)(data)
        def nth_feat(n):
            return ltn_gen_get_neigh_feats(m, nv, wnv)(data)[n]
        def fst_vec():
            return pack_vec(ltn_gen_get_main_vec(m, v, wv)(data))
        def nth_vec(n):
            return pack_vec(ltn_gen_get_neigh_vecs(m, nv, wnv)(data)[n])

        reducer = lambda outns, foo: outns

        (featmain, featneigh) = make_pipe("feat", fst_feat, nth_feat,
                                        self.dropout, config, reducer, m, h)
        (vecmain, vecneigh) = make_pipe("vec", fst_vec, nth_vec,
                                       self.dropout, config, reducer, m, h)

        zipped = [(featmain, vecmain)] + list(zip(featneigh, vecneigh))
        conc = list(map(lambda x: tf.concat(x, axis=1), zipped))
        cell = tf.nn.rnn_cell.LSTMCell(81, activation="linear") # HACK
        out, state = tf.nn.static_rnn(cell=cell, inputs=conc, dtype=tf.float32, scope="OUT")
        self.model = out[-1]

        var_list = [v for v in tf.trainable_variables() if
                    v.name.split('/')[0] in (["FCZfeat", "FCZvec", "FCfeat",
                                              "FCvec", "OUT", "rnnfeat", "rnnvec"])]


        kernels_by_shape = [["FCZfeat/kernel:0", "FCfeat/kernel:0"], ["FCZvec/kernel:0", "FCvec/kernel:0"],
                            ["OUT/kernel:0"], ["rnnfeat/kernel:0"], ["rnnvec/kernel:0"], ["OUT/lstm_cell/kernel:0"]]

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

        # Train op
        with tf.name_scope("train"):
            # Get gradients of all trainable variables
            gradients = tf.gradients(self.loss_op, var_list)
            gradients = list(zip(gradients, var_list))
            optimizer = tf.train.RMSPropOptimizer(learning_rate=config["learning_rate"],
                                                  decay=config["decay_rate"])
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0)
            self.train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    def train_hook(self, sess, handle):
        return sess.run([self.train_op, self.loss_op],
                        feed_dict={self.handle: handle,
                                   self.dropout: self._parameters["dropout_rate"]})[1]
    def eval_psgl_hook(self, sess, handle):
        return sess.run([self.rawout, self.scaled, self.y,
                         self.loss_op], feed_dict={self.handle: handle,
                                                   self.dropout: 1.})
