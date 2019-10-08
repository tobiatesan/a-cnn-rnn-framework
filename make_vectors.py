
import os
import argparse
import gensim
import gensim.models
import nus_johnson
import numpy as np

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--embeddings', required=True, choices=['w2v', 'wnet', 'synth'])
  parser.add_argument('output_file', help="Output file", type=str)
  parser.add_argument('--NUS_directory',  required=True, help="Directory with (Johnson-style) NUS files")
  parser.add_argument('--vecs_path',  required=True, help="Directory containing wn2vec.txt and/or GoogleNews-vectors-negative300.bin.gz")
  parser.add_argument('--tau', help="Tau (see Johnson et al.) -- try 5000 if unsure", type=int, required=True)
  parser.add_argument('--width', default=850, type=int, help="Vector width")
  args = parser.parse_args()

  w2v_path = os.path.join(args.vecs_path, "GoogleNews-vectors-negative300.bin.gz")
  wnet_path = os.path.join(args.vecs_path, "wn2vec.txt")

  if (args.embeddings == "w2v" or args.embeddings == "wnet"):
      paths_dict = {"w2v": lambda: gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True),
                    "wnet": lambda: gensim.models.KeyedVectors.load_word2vec_format(wnet_path, binary=False)}

      dictionary = paths_dict[args.embeddings]()
      def tag_to_vec(tag):
         try:
            try:
               return dictionary.wv[tag]
            except KeyError:
               return None # ???
         except Exception as e:
            print(e)
            sys.exit()
  else:
      assert(args.embeddings == "synth_wnet")
      import pickle
      with open("wnet_synth_dictionary", 'rb') as cfile:
          dictionary = pickle.load(cfile)
          def tag_to_vec(tag):         
              try:
                 try:
                    return dictionary[tag]
                 except KeyError:
                    return None # ???
              except Exception as e:
                 print(e)
                 sys.exit()
  nus_file = os.path.join(args.NUS_directory, "NUS_data.mat")
  imgids_file = os.path.join(args.NUS_directory, "imgids.mat")
  nus_object = nus_johnson.NUS_builder(nus_file, imgids_file)                 
  from make_fat_minimatrix import make_mini_fat_tag_matrix
  (mini_fat_tag_matrix, top_tau_indices) = make_mini_fat_tag_matrix(nus_object.nus_tag_matrix, args.tau)
  mini_tag_matrix = mini_fat_tag_matrix[:,:-1]
  tags_lists = [nus_object.nus_tags[x] for x in [top_tau_indices[np.nonzero(x)[1]] for x in mini_tag_matrix]]
  # We get a list of plain text words for each image, which...
  print("Will write to " + str(args.output_file))
  vectors = [np.sum(
      # ...we convert to vectors via tag_to_vec and then sum together for each image
      list(filter(lambda x: x is not None,
                  [tag_to_vec(x[0]) for x in filter(lambda z: z != "", y)]))
                  # Weirdness in the source files
      + [np.zeros([args.width,])],
      axis=0)
                 for y in tags_lists]
  np.save(args.output_file, vectors)
  print("Saved " + str(args.output_file))
