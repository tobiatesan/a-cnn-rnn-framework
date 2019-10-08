import os
import argparse
import numpy as np
import scipy.io
import nus_johnson
from scipy.sparse import hstack

'''
This program builds an adjacency matrix reduced to the top tau keywords (hence mini)
plus a column with original ids for each file on the side (hence fat)
'''

def make_mini_fat_tag_matrix(fat_tag_matrix, tau):
   tag_matrix = fat_tag_matrix[:,:-1]
   tag_frequency = np.sum(tag_matrix, axis=0)
   top_list = np.flip(np.argsort(tag_frequency))
   top_tau_indices = (np.array(top_list[:,0:tau])[0])
   top_tau_indices_plus_fatcol = np.concatenate([top_tau_indices, [tag_matrix.shape[1]]])
   mini_fat_tag_matrix = fat_tag_matrix[:,top_tau_indices_plus_fatcol]
   return (mini_fat_tag_matrix, top_tau_indices)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('split_file', help="Which split extract out of the .mat files", type=str)
  parser.add_argument('output_file', help="Output file", type=str)
  parser.add_argument('--tau', help="Tau (see Johnson et al.) -- try 5000 if unsure", type=int, required=True)
  parser.add_argument('--NUS_directory',  required=True, help="Directory with (Johnson-style) NUS files")
  args = parser.parse_args()
  split_file = scipy.io.loadmat(args.split_file)
  split_file = split_file["split"][0]

  nus_file_path = os.path.join(args.NUS_directory, "NUS_data.mat")
  imgids_file_path = os.path.join(args.NUS_directory, "imgids.mat")
  nus_object = nus_johnson.NUS_builder(nus_file_path, imgids_file_path)

  # Numpy weirdness
  hack = np.reshape(range(nus_object.nus_tag_matrix.shape[0]), (-1,1))
  fat_matrix = hstack([nus_object.nus_tag_matrix, hack]).tocsr()
  
  (fat_mini_matrix, top_tau_indices) = make_mini_fat_tag_matrix(fat_matrix, args.tau)
  scipy.sparse.save_npz(args.output_file, fat_mini_matrix)
  print("Saved FAT minimatrix to ", args.output_file)
