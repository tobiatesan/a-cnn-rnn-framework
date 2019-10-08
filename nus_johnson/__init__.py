import numpy as np
import os
import scipy
import scipy.io
from .consts import *
import types

def _load_data(data_file, key):
   basen = os.path.basename(data_file)
   cachef = basen+".cache.npy"
   try:
      mat = np.load(cachef, 'r')
   except:
      print("Couldn't load", cachef)
      print("Loading", data_file)
      import h5py
      _file = h5py.File(data_file, 'r')
      _size = _file[key].shape[0]
      mat = np.array(
         [np.transpose(
            _file[x]
         ).astype('i1').tostring()
          for x in _file[key][:,0]])
      mat = np.reshape(mat, [_size, 1])
      print("Loaded", data_file)
      try:
         np.save(cachef, mat)
         print("Saved", cachef)
         mat = np.load(cachef, 'r')
         print("Reloaded", cachef)
      except:
         print("Couldn't save", cachef)
   return mat

def _load_features(data_file = "feat7_caffe_all.mat"):
   basen = os.path.basename(data_file)
   cachef = basen+".cache.npy"
   try:
      # print("Loading", cachef)
      mat = np.load(cachef, 'r')
      # print("Loaded", cachef)
   except:
      print("Couldn't load", cachef)
      print("Loading", data_file)
      import h5py
      import hdf5storage
      with h5py.File(data_file, 'r') as file:
         mat = np.array(file["feat7"])
      print("Loaded", data_file)
      try:
         np.save(cachef, mat)
         print("Saved", cachef)
         mat = np.load(cachef, 'r')
      except:
         print("Couldn't save", cachef)
   return mat

def NUS_builder(nus_file_path):
     # For performance reasons, mostly
     nus_imgids_file = _load_data(os.path.join(nus_file_path,"imgids.mat"), "img_ids")
     nus_imgids = np.reshape(nus_imgids_file, (consts["nfiles_original"]))
     nus_file = scipy.io.loadmat(os.path.join(nus_file_path, "NUS_data.mat"))
     nus_label_matrix = nus_file["NUS"]["label_matrix"][0,0]
     nus_tags = nus_file["NUS"]["tags"][0][0][0]
     nus_labels = nus_file["NUS"]["labels"][0][0][0]
     nus_tag_matrix = nus_file["NUS"]["tag_matrix"][0,0]
     nus_photo_ids = nus_file["NUS"]["photo_ids"][0][0][0]
     nus_tags = nus_file["NUS"]["tags"][0][0][0]
     features = NUS_Features(os.path.join(nus_file_path, "feat_alexnet",  "feat7_caffe_all.mat"))
     nus = NUS(nus_label_matrix, nus_tag_matrix, nus_imgids, nus_photo_ids, nus_tags)
     return (nus, features)

class NUS_Features:
   def __init__(self, features_file_path):
      self.features = _load_features(features_file_path)
   def get_features_for_bigid(self, bigid):
      return self.features[bigid,:]


class NUS:
   def __init__(self, nus_label_matrix, nus_tag_matrix, nus_imgids, nus_photo_ids, nus_tags):
      self.nus_label_matrix = nus_label_matrix
      self.nus_imgids = nus_imgids
      self.nus_tag_matrix = nus_tag_matrix
      self.nus_photo_ids = nus_photo_ids
      self.nus_tags = nus_tags
      
   def get_labels_for_matrixid(self, matrixid):
      return np.reshape(np.array(self.nus_label_matrix[matrixid].todense()), (consts["nlabels"]))

   def get_flickrtags_for_matrixid(self, matrixid):
      return np.reshape(self.np.array(nus_tag_matrix[matrixid].todense()), (consts["ntags"]))

   def get_flickrid_for_bigid(self, bigid):
      return self.nus_imgids[bigid].astype(np.int)
        
   def get_flickrid_for_matrixid(self, matrixid):
      return self.nus_photo_ids[matrixid][0].astype(np.int)

   def get_bigid_for_flickrid(self, flickrid):
      return np.argwhere(self.nus_imgids.astype(np.int) == flickrid)[0][0]
