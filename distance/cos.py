import scipy.spatial.distance
import numpy as np

def M_nearest(_id, fat_vectors, M, train_mask):
   return find_w2v_for_id(_id, fat_vectors, train_mask)[0:M]

def cos_cdist(matrix, vector):
    """
    Compute the cosine distances between each row of matrix and vector.
    """
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)

def find_w2v_for_id(_id, fat_vectors, train_mask):
   vectors = fat_vectors[:,:-1]
   vec = vectors[_id]
   vec = vec.reshape(1, -1)
   allowable_fat_vectors = fat_vectors[train_mask]
   allowable_vectors = allowable_fat_vectors[:,:-1]
   allowable_vektors = np.matrix(allowable_vectors)
   sortd = np.argsort(cos_cdist(allowable_vektors, vec))
   sortd = sortd[1:] # ignore x itself
   allowable_fat_results = allowable_fat_vectors[sortd]
   allowed_indices = allowable_fat_results[:,-1]
   return allowed_indices
