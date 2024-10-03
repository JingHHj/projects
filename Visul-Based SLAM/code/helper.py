import numpy as np
import pr3_utils as utils
import scipy as sp
import torch

def find_unobserved(landmarks):
	"""
	find all the unobserved landmarks [-1, -1, -1, -1] for every time step
	output a matrix in bool type for landmark in every time steps
	"""
	pattern = np.array([-1, -1, -1, -1])
	idx = np.empty(landmarks.shape[1:], dtype = bool)
	for i in range(landmarks.shape[2]):
		# for every timestep if theres a land mark is unobserved, export false
		idx[:,i] = np.all(landmarks[:,:,i] != pattern.reshape(-1, 1), axis=0)
	# landmark_t = landmarks[:,idx[:,10],10]	
	return idx

def get_homogeneous(matrix):
	"""
	input normal matrix 
	return matrix in homogenous form
	"""
	match len(matrix.shape):
		case 2:
			ones = np.ones((1,matrix.shape[1]))
		case 3:
			ones = np.ones((1,matrix.shape[1],matrix.shape[2]))
		case _:
			print("Wrong matrix dimension")
			return None
	
	homo_matrix = np.vstack((matrix, ones))
	return homo_matrix

def change_features_form(target,k = int):
	"""
	input features in [xl,xr,yl,yr] form and original numbers (4, 13289, 3026)
	output with [xl,yl,xr,yr] form and with a ratio k
	"""
	result = np.copy(target)
	result[1,:,:] = target[2,:,:]
	result[2,:,:] = target[1,:,:]
	return result[:,::k,:]



def get_stereo_ks(matrix, baseline):
	"""
	input the extrinsic matrix and the baseline
	output its form for stereo camera
	"""
	m = np.hstack((
		np.vstack((matrix[:2,:],matrix[:2,:])),
		np.array([0.,0.,-K[0,0]*baseline,0.]).reshape(4,1)
		))
	return m


def circle_dot(vector):
	"""
	matrix: N*4 of N homogenous form vector
	output: 
		[I, -s_hat]
		[0,    0  ]
		4*6
	"""
	if vector.shape[1] > 4:
		vector = np.moveaxis(vector,1,0)
	
	result = np.zeros(vector.shape[:-1]+(4,6),dtype = float)
	result[...,:3,:3] = np.eye(3)
	result[...,:3,3:] = - utils.axangle2skew(vector[:,:3])
	return result
