import os
import numpy as np
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.data import Iterator

def make_dataset(preprocessed_array, features_object, minimatrix, embeddings, config, embeddings_width, test_mode=False):
    n = int(config["n"])
    m = int(config["m"])
    M = int(config["M"])
    import itertools
    from nus_johnson.types import feat_dtype, truth_dtype, feat_shape, truth_shape, ltn_gen_type
    
    v = 0
    nv = 0
    wv = 0
    wnv = 0

    if (config["arch"] in ["ltn+vecs", "ltn+allvecs"]):
        if (embeddings is None):
            assert(minimatrix is not None)
            v = 1
        else:
            assert(minimatrix is None)
            wv = 1
            
    if (config["arch"] in ["ltn+allvecs"]):
        if (embeddings is None):
            assert(minimatrix is not None)
            nv = 1
        else:
            assert(minimatrix is None)
            wnv = 1

    if (config["arch"] in ["ltwin", "ltwin+rnn", "ltwin+2rnn"] or config["arch"] == "lzip"):
        if (embeddings is not None):
            wv = wnv = 1
        else: 
            v = nv = 1
    
    get_vector = lambda x: list()
    get_w2v_vector = lambda x: list()
    if (v + nv > 0):
        assert(minimatrix is not None)
        print(minimatrix.shape)
        get_vector = lambda nusid: np.array(minimatrix[nusid].todense()).ravel()
    if (wv + wnv > 0):
        assert(embeddings is not None)
        print("Feeding transformed embeddings from file")
        get_w2v_vector = lambda nusid: embeddings[nusid]
 
    print("SWITCHES:", v, nv, wv, wnv)

    def gen():
        import nus_johnson.consts
        import random
        allcombinations = list(itertools.combinations(range(M), m)) # Precompute this
        for i in range(preprocessed_array.shape[0]):
            row = preprocessed_array[i]
            bigid = row["bigid"]
            truth = row["truth"]
            feature = features_object.get_features_for_bigid(bigid)
            found = row["found_neighbors"]
            neighbors = list(row["neighbors"][0:found])
            if (len(neighbors) < M):
                neighbors = neighbors + random.sample(range(nus_johnson.consts["nfiles"]), (M - len(neighbors)))
            assert(len(neighbors) >= M)
            neighbors = neighbors[0:M]
            assert(len(neighbors) == M)
            vector = get_vector(row["nusid"])
            if (wv + wnv):
                w2v_vector = get_w2v_vector(row["nusid"])
            if (not test_mode):
                random_c = random.sample(neighbors, m)
                feature_c = list(map(lambda x : features_object.get_features_for_bigid(x), random_c))
                additional = []
                if (nv):
                    vector_c = list(map(get_vector, random_c))
                    additional = additional + [vector] + vector_c
                if (wnv):
                    w2v_vector_c = list(map(get_w2v_vector, random_c))
                    additional = additional + [w2v_vector] + w2v_vector_c
                yield tuple([feature] + feature_c + [truth] + additional)

            else:
                if (len(allcombinations) <= n):
                    assert(len(allcombinations) == 1)
                    for i in range(n):
                        combi = allcombinations[0]
                        random_c = [neighbors[x] for x in combi]
                        feature_c = list(map(lambda x : features_object.get_features_for_bigid(x), random_c))
                        additional = []
                        if (nv):
                            vector_c = list(map(get_vector, random_c))
                            additional = additional + [vector] + vector_c
                        if (wnv):
                            w2v_vector_c = list(map(get_w2v_vector, random_c))
                            additional = additional + [w2v_vector] + w2v_vector_c
                        yield tuple([feature] + feature_c + [truth] + additional)
                else:
                    for combi in random.sample(allcombinations, n):
                        random_c = [neighbors[x] for x in combi]
                        feature_c = list(map(lambda x : features_object.get_features_for_bigid(x), random_c))
                        additional = []
                        if (nv):
                            vector_c = list(map(get_vector, random_c))
                            additional = additional + [vector] + vector_c
                        if (wnv):
                            w2v_vector_c = list(map(get_w2v_vector, random_c))
                            additional = additional + [w2v_vector] + w2v_vector_c
                        yield tuple([feature] + feature_c + [truth] + additional)

    return tf.data.Dataset.from_generator(gen, ltn_gen_type(m, v, nv, wv, wnv))


def train_and_test(sess, trainset, validationset, testset, minimatrix, embeddings, config):
    from nus_johnson.types import feat_dtype, truth_dtype, truth_shape

    from nus_johnson.types import feat_shape

    if (config["arch"] in ["ltn", "rtn", "ltn+vecs", "ltn+allvecs"]):
        from models.ltn import ltn as network
    else:
        if (config["arch"] in ["ltwin", "ltwin+rnn", "ltwin+2rnn"]):
            from models.ltwin import ltwin as network
        else:
            if(config["arch"] == "lzip"):
                from models.ltwin import ltwin as network
            else:
                print("Unrecognized arch, check your configuration file")
                assert(False)

    from trainer.evaluation import evaluate
    from trainer import Trainer
    n = int(config["n"])
    m = int(config["m"])

    def preprocess(results):
        print("Preprocessing results with n = ", n)
        nochunks = results.shape[0]/n
        chunked = np.array_split(results, nochunks, axis=0)
        new_res = list()
        for chunk in chunked:
            assert(np.all((chunk[0,:] == chunk)[:,2,:]))
            reduced = np.mean(chunk, axis=0)
            new_res.append(reduced)
            assert(np.all(chunk[0,2] == reduced[2]))
        new_res = np.array(new_res, results.dtype)
        pred = new_res[:,0,:]
        pred = np.round(pred)
        new_res[:,0,:] = pred
        return new_res


    training_iterator = trainset.make_initializable_iterator()
    validation_iterator = validationset.make_initializable_iterator()
    test_iterator = testset.make_initializable_iterator()

    input_types = (feat_dtype, truth_dtype)
    input_shapes = (feat_shape, truth_shape)
    tau = -1
    if (minimatrix is not None):
        tau = (minimatrix.shape[1])
    embeddings_width = -1
    if (embeddings is not None):
        embeddings_width = (embeddings.shape[1])
    model = network(input_types, input_shapes, m, tau, embeddings_width, config)
    sess.run(tf.global_variables_initializer())

    trainer = Trainer(training_iterator, validation_iterator, test_iterator, config=config, preprocess=preprocess)
    trainer.train(model, sess, int(config["num_epochs"]))
    results = trainer.test(model, sess)
    return results
