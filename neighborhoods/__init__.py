import multiprocessing
import numpy as np
import nus_johnson.types

def delta_format(delta: np.timedelta64) -> str:
    days = delta.astype("timedelta64[D]") / np.timedelta64(1, 'D')
    hours = int(delta.astype("timedelta64[h]") / np.timedelta64(1, 'h') % 24)
    if days > 0 and hours > 0:
        return "{} d, {} h".format(days,hours)
    elif days > 0:
        return "{} d".format(days)
    else:
        return "{} h".format(hours)

def make_neighborhoods(x):
       import os
       vector_mode = x[-1]
       if vector_mode:
          (train_idxs, max_neighborhood_size, train_ids, orig_ids, train_mask, vectors, nus, _) = x
       else:
          (train_idxs, max_neighborhood_size, train_ids, orig_ids, train_mask, fat_minimatrix, nus, _) = x

       if vector_mode:
           idxs = np.reshape(range(vectors.shape[0]), (-1,1))
           fat_vectors = np.concatenate([vectors, idxs], axis=1)
       else:
           pass           
           
       rang = range(train_idxs.shape[0])
       my_array = np.empty(dtype=nus_johnson.types.precalc_dtype(max_neighborhood_size), shape=(len(rang)))
       import traceback
       start = np.datetime64('now')
       for q in rang:
          try:
                 i = train_idxs[q]
                 matrixid = orig_ids[train_ids[i]-1]-1
                 flickrid = nus.get_flickrid_for_matrixid(matrixid)
                 onehot = nus.get_labels_for_matrixid(matrixid)
                 bigid = nus.get_bigid_for_flickrid(flickrid)
                 my_array[q]["bigid"] = bigid
                 my_array[q]["nusid"] = matrixid
                 my_array[q]["truth"] = onehot
                 if vector_mode:
                    import distance.cos
                    candidate = distance.cos.M_nearest(matrixid, fat_vectors, max_neighborhood_size, train_mask)
                 else:
                    import distance.jaccard 
                    candidate = distance.jaccard.M_nearest(matrixid, fat_minimatrix, max_neighborhood_size, train_mask)

                 try: # np weirdness
                     candidate = np.reshape(np.array(candidate.todense()),(-1))
                 except AttributeError:
                     candidate = np.reshape(np.array(candidate),(-1))
                     
                 my_array[q]["found_neighbors"] = candidate.shape[0]
                 candidate = np.pad(candidate, (0,max_neighborhood_size-candidate.shape[0]), 'constant', constant_values = 0)
                 my_array[q]["neighbors"] = candidate
                 if (q%50 == 0):
                    time = np.datetime64('now')
                    ratio = float(q)/float(train_idxs.shape[0])
                    try:
                       print("{}: Worker: {} Done {}%, running {}, remaining {}".format(
                          str(time) ,
                          multiprocessing.current_process(),
                          ratio*100,
                          delta_format(time-start),
                          delta_format((time-start)*(float(1-ratio)/float(ratio))),
                          ))
                    except ZeroDivisionError:
                       print("{}: Worker: {} Done {}%".format(                           
                          str(time),
                          multiprocessing.current_process(),
                          ratio*100,
                          ))

          except Exception as e:
                 print(str(e))
                 print(traceback.format_exc())
                 print(traceback.format_exc(e))
       print("DONE!")
       print(i)
       return my_array
