import pickle
import sys
import time 
import jax.numpy as jnp
import numpy as np

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

# cfile = "../data/cam/cam" + dataset + ".p"
# ifile = "../data/imu/imuRaw" + dataset + ".p"
# vfile = "../data/vicon/viconRot" + dataset + ".p"



 # train data
dataset="9"
cfile = "../data/trainset/cam/cam" + dataset + ".p"
ifile = "../data/trainset/imu/imuRaw" + dataset + ".p"
vfile = "../data/trainset/vicon/viconRot" + dataset + ".p"

#   # test data
# dataset="11"
# cfile = "../data/testset/cam/cam" + dataset + ".p"
# ifile = "../data/testset/imu/imuRaw" + dataset + ".p"
# # vfile = "../data/trainset/vicon/viconRot" + dataset + ".p"


vicd = read_data(vfile)
ts = tic()
camd = read_data(cfile)
# (240, 320, 3, 1685) v,h
imud = read_data(ifile)

toc(ts,"Data import")



# tau_imud = jnp.array(np.diff((imud['ts']))).transpose() # ()
#     # get the tau,which is the differences between the current time step and next time step
# raw_imud = jnp.array(imud['vals']).transpose()
#     # get the a and w from the origial data








