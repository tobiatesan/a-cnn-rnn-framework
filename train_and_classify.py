import argparse
import os
import numpy as np
from datetime import datetime
from nus_johnson import NUS_builder

#### Limit RAM usage ####

import resource

MAX_RAM_MEGS = 6 * 1024

rsrc = resource.RLIMIT_DATA
soft, hard = resource.getrlimit(rsrc)
print('Soft limit starts as  :', soft)

resource.setrlimit(rsrc, (MAX_RAM_MEGS * 1024 * 1024, hard))

soft, hard = resource.getrlimit(rsrc)
print("Using max {} bytes of RAM".format(MAX_RAM_MEGS * 1024 * 1024))

#######

ALLOW_GROWTH = True
GPU_MEMORY_FRACTION = .25

#####################

import os
import numpy as np

def make_dirs(directory):
    if (len(directory) > 0):
        if not os.path.exists(directory):
            os.makedirs(directory)

def run(
                 files_triple,
                 output_file_name,
                 minimatrix,
                 embeddings,
                 features_object,
                 config):
   embeddings_width = -1 
   if embeddings is not None:
       embeddings_width = embeddings.shape[1]
       print("Autodetected embedding width", embeddings_width)
   from experiments.ltn import make_dataset, train_and_test
   (training_input_file, test_input_file, validation_input_file) = files_triple
   results = []
   i = 0
   import tensorflow as tf
   print("Loading", training_input_file, validation_input_file, test_input_file)
   print ("Names at load", training_input_file, validation_input_file, test_input_file)
   train_array = np.load(training_input_file)
   validation_array = np.load(validation_input_file)
   test_array = np.load(test_input_file)
   print ("Shapes at load", train_array.shape, validation_array.shape, test_array.shape)
   tf.reset_default_graph()
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION, allow_growth=ALLOW_GROWTH)
   with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
       trainset = make_dataset(train_array, features_object, minimatrix, embeddings, config, embeddings_width)
       validationset = make_dataset(validation_array, features_object, minimatrix, embeddings, config, embeddings_width, test_mode=True)
       testset = make_dataset(test_array, features_object, minimatrix, embeddings, config, embeddings_width, test_mode=True)

       trainset = trainset.\
                  batch(int(config["train_batch_size"])).\
                  prefetch(int(config["train_prefetch"]))
       validationset = validationset.\
                       batch(int(config["test_batch_size"])).\
                       prefetch(int(config["test_prefetch"]))
       testset = testset.\
                 batch(int(config["test_batch_size"])).\
                 prefetch(int(config["test_prefetch"]))
       res = train_and_test(sess, trainset, validationset, testset, minimatrix, embeddings, config)
       make_dirs(os.path.dirname(output_file_name))
       np.save(output_file_name, res)
       print ("Done, saved in", output_file_name)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('config_file', help="Configuration to run", type=str)
  parser.add_argument('training_input_file', help="Input file with neighborhoods for training", type=str)
  parser.add_argument('validation_input_file', help="Input file with neighborhoods", type=str)
  parser.add_argument('test_input_file', help="Input file with neighborhoods for test", type=str)
  parser.add_argument('output_file', help="Output file", type=str)
  parser.add_argument('--NUS_directory',  required=True, help="Directory with (Johnson-style) NUS files")
  parser.add_argument('--semantic_embeddings', type=str, help="File with precomputed semantic vectors, mutually exclusive with fat_minimatrix")
  parser.add_argument('--fat_minimatrix_file',  help="(Fat) minimatrix, required iff using jaccard, mutually exclusive with semantic embeddings")
  args = parser.parse_args()

  (_, features_object) = NUS_builder(args.NUS_directory)

  import json
  with open(args.config_file) as json_data:
      config = json.load(json_data,)

  minimatrix = None
  if (args.fat_minimatrix_file is not None):
    assert(args.semantic_embeddings is None)
    import scipy
    fat_mini_matrix = scipy.sparse.load_npz(args.fat_minimatrix_file)
    print(fat_mini_matrix.shape)
    print("Unfattening minimatrix")
    minimatrix = fat_mini_matrix[:,:-1]

  embeddings = None
  embeddings_width = -1
  
  if (args.semantic_embeddings is not None): 
    assert(args.minimatrix is None)
    print("Loading ",args.semantic_embeddings)
    embeddings = np.load(args.semantic_embeddings)

  run((args.training_input_file, args.test_input_file, args.validation_input_file), args.output_file, minimatrix, embeddings, features_object, config)
