import numpy as np
from .consts import *

feat_dtype = feature_dtype = np.float32
bigid_dtype = np.uint32
nusid_dtype = np.uint32
truth_dtype = np.uint8
neighbors_dtype = bigid_dtype
filename_dtype = np.dtype('a47')
foundneighbors_dtype = np.uint8
MAX_NEIGHBORHOOD_SIZE = 24

feat_shape = (None, 4096)
truth_shape = (None, consts["nlabels"])

image_dtype = np.float32
image_shape = (None, 227, 227, 3)
vgg_image_shape = (None, 224, 224, 3)

vector_dtype = np.uint8
vector_shape = lambda tau: (None, tau)

w2v_vector_dtype = np.float64
w2v_vector_shape = lambda x: (None, x) # check



def precalc_dtype (neighborhood_size):
    return np.dtype(
           [('bigid', bigid_dtype),
            ('nusid', nusid_dtype),
            ('truth', truth_dtype, consts["nlabels"]),
            ('found_neighbors', foundneighbors_dtype),
            ('neighbors', neighbors_dtype, neighborhood_size)
           ])


ltn_gen_type = lambda m, v, nv, wv, wnv: tuple([feat_dtype] +
                                               [feat_dtype]*m +
                                               [truth_dtype] +
                                               [vector_dtype]*v +
                                               [vector_dtype]*m*v*nv +
                                               [w2v_vector_dtype]*wv +
                                               [w2v_vector_dtype]*m*wv*wnv)

ltn_gen_shape = lambda vector_width: lambda m, v, nv, wv, wnv, tau: tuple([feat_shape] +
                                                     [feat_shape]*m +
                                                     [truth_shape] +
                                                     [vector_shape(tau)]*v +
                                                     [vector_shape(tau)]*m*v*nv +
                                                     [w2v_vector_shape(vector_width)]*wv +
                                                     [w2v_vector_shape(vector_width)]*m*wv*wnv)

ltn_gen_get_main_feat = lambda m, v, wv: lambda vec: vec[0]
ltn_gen_get_neigh_feats = lambda m, v, wv: lambda vec: vec[1:m+1]
ltn_gen_get_truth = lambda m, v, wv: lambda vec: vec[m+1]
ltn_gen_get_main_vec = lambda m, v, wv: lambda vec: vec[m+2]
ltn_gen_get_neigh_vecs = lambda m, v, wv: lambda vec: vec[m+2+1:m+2+m+1]
ltn_gen_get_main_w2vec = lambda m, v, wv: lambda vec: vec[m+2+m+1]
ltn_gen_get_neigh_w2vecs = lambda m, v, wv: lambda vec: vec[m+2+m+1:m+2+m+1+m]

ltn_gen_set_main_feat = lambda m, v, wv: lambda vec, x: tuple([x]) + vec[1:]
ltn_gen_set_neigh_feats = lambda m, v, wv: lambda vec, l: vec[:1] + tuple(l) + vec[m+1:]
ltn_gen_set_truth = lambda m, v, wv: lambda vec, x: vec[:m+1] + tuple([x]) + vec[m+2:]
ltn_gen_set_main_vec = lambda m, v, wv: lambda vec, x: vec[:m+2] + tuple([x]) + vec[m+3:]
ltn_gen_set_neigh_vecs = lambda m, v, wv: lambda vec, l: vec[:m+2+1] + tuple(l)
ltn_gen_set_main_w2vec = lambda m, v, wv: lambda vec, x: vec[:m+2+m] + tuple([x]) + vec[m+3+m:]
ltn_gen_set_neigh_w2vecs = lambda m, v, wv: lambda vec, l: vec[:m+2+1+m] + tuple(l)
