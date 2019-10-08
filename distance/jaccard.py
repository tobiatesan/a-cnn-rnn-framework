import numpy as np

def jaccard_distance(vec, mat):
    intersection = np.sum(
       np.multiply(mat, np.reshape(vec, (1, -1))), axis = 1)
    a = np.sum(mat, axis = 1)
    b = np.sum(vec)
    # distance = 1 - similarity
    res = 1 - (intersection / (a + b - intersection))
    return res

def find_nn_for_id(_id, fat_tag_matrix, train_mask):
   tag_matrix = fat_tag_matrix[:,:-1]
   tags_for_id = np.nonzero(tag_matrix[_id,:])[1]
   ids_of_tag_neighbours = np.nonzero(tag_matrix[:,tags_for_id])[0]
   # one neighbour can be neighbour on count of MORE than one tag
   ids_of_tag_neighbours = np.unique(ids_of_tag_neighbours)
   # remove image from its own neighborhood
   ids_of_tag_neighbours = ids_of_tag_neighbours[np.nonzero(ids_of_tag_neighbours != _id)[0]]
   # use only train set as neighbours
   ids_of_tag_neighbours = np.intersect1d(ids_of_tag_neighbours, train_mask)
   id_array = np.array(tag_matrix[_id].todense())[0]
   tags_for_neighbours = tag_matrix[ids_of_tag_neighbours,:].todense()
   if (tags_for_neighbours.shape[0] == 0):
      print("WARNING: No neighbours for", _id)
      return fat_tag_matrix[
         np.random.random_integers(0, fat_tag_matrix.shape[0], 100)
         ,-1]
   scores = np.transpose(jaccard_distance(id_array, tags_for_neighbours))[0]
   # argsort means small to big, so closest are least distant, so closer
   rows_of_closest_neighbours = ids_of_tag_neighbours[np.argsort(scores, kind="heapsort")][0]
   return fat_tag_matrix[rows_of_closest_neighbours,-1]

def M_nearest(_id, tag_matrix, M, train_mask):
   return find_nn_for_id(_id, tag_matrix, train_mask)[0:M]

