import argparse
import nus_johnson
import scipy
import scipy.io
import numpy as np
import multiprocessing as multi
import neighborhoods
import os
import sys

'''
This program takes data files in the format of Johnson et al and pre-computes neighbourhoods
'''


DEFAULT_THREADS = 4

try:
    import psutil
    import math
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    THREADS = int(math.ceil(float(cpu_cores + cpu_threads) / 2.0))
    print("Using auto threads: ", THREADS)
except Exception as e:
    THREADS = DEFAULT_THREADS
thread_pool = multi.Pool(THREADS)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('split_file', help="Which split extract out of the .mat files", type=str)
  parser.add_argument('output_file', help="Output file", type=str)
  parser.add_argument('--max_neighborhood_size', type=int, required=True)
  parser.add_argument('--closed_neighborhoods', help="Set to false to replicate Johnson et al")  
  parser.add_argument('--NUS_directory',  required=True, help="Directory with (Johnson-style) NUS files")
  parser.add_argument('--phase', required=True, help="Process train, test or validation?", type=str, choices=["train", "test", "validation"])
  parser.add_argument('--vector_mode', help="Use vectors+cos, otherwise plain Jaccard", action="store_true")
  parser.add_argument('--embedding_vectors_file', type=str, help="File with precomputed vectors, required iff using vector mode")
  parser.add_argument('--fat_minimatrix_file',  help="Fat minimatrix, required iff using jaccard")
  args = parser.parse_args()
  split_file = scipy.io.loadmat(args.split_file)
  split = split_file["split"][0] 
  assert(split["train_id"][0].shape[1] == nus_johnson.consts["ntrain"]), str(split["train_id"][0].shape)+" "+str(nus_johnson.consts["ntrain"])
  if (args.vector_mode):
      if (args.embedding_vectors_file is None):
          print("Specify --embedding_vectors_file")
          sys.exit()
      either_vectors_or_minimatrix = np.load(args.embedding_vectors_file)
  else:
      if (args.fat_minimatrix_file is None):
          print("Specify --fat_minimatrix_file")
          sys.exit()
      either_vectors_or_minimatrix = scipy.sparse.load_npz(args.fat_minimatrix_file)
  
  nus_file = os.path.join(args.NUS_directory, "NUS_data.mat")
  imgids_file = os.path.join(args.NUS_directory, "imgids.mat")
  nus_object = nus_johnson.NUS_builder(nus_file, imgids_file)

  '''
  Matlab files used by Johnson & al for their experiments use
  different ids for a given image than standard NUS-Wide for
  performance reasons This is the key matching "regular" NUS-Wide ids
  with their own
  '''  
  orig_ids = np.reshape(split["orig_ids"][0], (split["orig_ids"][0].shape[0]))

  
  if (args.closed_neighborhoods): 

    train_mask_for_neighs = (np.reshape(
                split["train_id"][0],
                (nus_johnson.consts["ntrain"],)
            )[0:nus_johnson.consts["ntrain_resplit"]])
    train_mask_for_neighs = np.array(orig_ids[train_mask_for_neighs-1]-1)

    validation_mask_for_neighs = (np.reshape(
                split["train_id"][0],
                (nus_johnson.consts["ntrain"],)
            )[nus_johnson.consts["ntrain_resplit"]:])
    validation_mask_for_neighs = np.array(orig_ids[validation_mask_for_neighs-1]-1)

    test_mask_for_neighs = (np.reshape(
                split["test_id"][0],
                (nus_johnson.consts["ntest"],)
            ))
    test_mask_for_neighs = np.array(orig_ids[test_mask_for_neighs-1]-1)
  else:
    # Use everything  
    train_mask_for_neighs = np.array(orig_ids-1)
    validation_mask_for_neighs = np.array(orig_ids-1)
    test_mask_for_neighs = np.array(orig_ids-1)

  job_tuple = {
    "train": (np.reshape(split["train_id"][0], (nus_johnson.consts["ntrain"]))[:nus_johnson.consts["ntrain_resplit"]],
          train_mask_for_neighs),
    "validation": (np.reshape(split["train_id"][0], (nus_johnson.consts["ntrain"]))[nus_johnson.consts["ntrain_resplit"]:],
          validation_mask_for_neighs),
    "test": (np.reshape(split["test_id"][0], (nus_johnson.consts["ntest"],)),
          test_mask_for_neighs)}
  

  (split_half_ids, mask_for_neighs) = job_tuple[args.phase]
  # We split the images into chunks and call the nn function via the multiprocessing module -- purely for performance reasons
  chunks = np.array_split(range(split_half_ids.shape[0]), THREADS)
  args2 = [(chunk, args.max_neighborhood_size, split_half_ids, orig_ids, mask_for_neighs, either_vectors_or_minimatrix, nus_object,
           args.vector_mode) for chunk in chunks]
  multiple_results = thread_pool.map(neighborhoods.make_neighborhoods, args2)
  precalc_array = np.concatenate(multiple_results, axis=0)
  np.save(args.output_file, precalc_array)
  print("Saved " + args.output_file)
