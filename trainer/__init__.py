from datetime import datetime
import tensorflow as tf
from trainer.evaluation import evaluate

class Trainer(object):
    def __init__(self,
                 train_iterator,
                 validation_iterator,
                 test_iterator,
                 config=None,
                 preprocess=None):
        self.train_iterator = train_iterator
        self.validation_iterator = validation_iterator
        self.test_iterator = test_iterator
        self.preprocess = preprocess
        self.config = config

    def _pre_epoch(self, sess, epoch):
        sess.run(self.train_iterator.initializer)
        
    def _eval_and_save(self, model, epoch, sess, saver, pre_test_hook, handle):
        pre_test_hook(sess)
        (avgloss, _) = evaluate(model, sess, handle, self.preprocess)
        self._save(sess)
        return (avgloss, 0, 0, 0, 0)

    def _pre_test(self, sess):
        sess.run(self.test_iterator.initializer)
        
    def save(self):
        from trainer.evaluation import test
        full_results = evaluate(model, sess, sess.run(self.test_iterator.string_handle()), self.preprocess)

    def _test(model, sess, handle, preprocess = None):
        test_loss = 0
        test_count = 0
        full_results = None
        while True:
            try:
                predictions, scores, ground, loss = model.eval_psgl_hook(sess, handle)
                test_count += 1
                test_loss += loss
                pack = np.stack((predictions, scores, ground), axis = 1)
                if (full_results is None):
                    full_results = pack
                else:
                    full_results = np.concatenate((full_results,
                                                   pack), axis = 0)
            except tf.errors.OutOfRangeError:
                break
        if (preprocess is not None):
            full_results = preprocess(full_results)
        return full_results

    def test(self, model, sess):
        self._pre_test(sess)
        (_, full_results) = evaluate(model, sess, sess.run(self.test_iterator.string_handle()), self.preprocess)
        return full_results

    def _post_epoch(self, model, sess, count, epoch):
        def pre_validation_hook(sess):
            sess.run(self.validation_iterator.initializer)
        return self._eval_and_save(model, epoch, sess, self.saver, pre_validation_hook, sess.run(self.validation_iterator.string_handle()))

    def _batch(self, model, sess, epoch, second_pass=False):
        loss = model.train_hook(sess, sess.run(self.train_iterator.string_handle()))
        (self.count, self.globcount) = (self.count + 1, self.globcount + 1)
        return loss

    def train(self, model, sess, num_epochs, second_pass=False):

        ###################################################################
        # Begin training
        ###################################################################
        self.globcount = 0
        for epoch in range(num_epochs):
            ###################################################################
            # Actual training logic
            ###################################################################

            self.count = 0
            self._pre_epoch(sess, epoch)
            try:
                while True:
                    loss = self._batch(model, sess, epoch)
                    if (self.count % 100 == 0):
                        print("Done batch", self.count, "training loss: ", loss)
            except tf.errors.OutOfRangeError:
                # Reached the end of dataset
                pass
        return (self.globcount, epoch, loss)
