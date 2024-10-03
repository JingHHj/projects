import numpy as np
from pr3_utils import *
import pr3_utils as utils
import scipy as sp
import plot
import torch
import helperfunction as hp


def imu_prediction_init(twists):
	"""
	initialize for imu prediction step
	input:
		twist: twists in for of (6,n) or (n,6)
	output:
		mean: the initialized total mean matrix of imu prediztion (3026, 4, 4)
		cov: the initialized total mean matrix of imu prediztion  (3026, 6, 6)
		u4: twists in (4,4) form (3026, 4, 4)
		u6: twists in (6,6) form (3026, 6, 6)
	"""
	if twists.shape[0] == 6:
		twists = twists.T
		# make sure the twists are in the form of n*6	
	u4 = utils.axangle2twist(twists) # (3026, 4, 4)
	u6 = utils.axangle2adtwist(twists) # (3026, 6, 6)

	mean = np.tile(np.eye(4),(u4.shape[0],1,1)) # 3026*4*4
	cov_0 = np.diag([0.1,0.1,0.1,0.1,0.1,0.1]) # initialize cov 
	cov = np.tile(cov_0,(u4.shape[0],1,1)) # 3026*6*6
	return mean,cov,u4,u6

def imu_prediction(mean_current,cov_current,time,ut4,ut6):
	"""
	input:
		mean_current: the imu mean(pose) of current time step i (4,4)
		cov_current: the imu cov of current time step i (6,6)
		time: current time interval
		ut4: the twists in (4,4) at current step
		ut6: the twists in (6,6) at current step
	return:
		mean: the prediction step result of imu mean (4,4)
		cov: the prediction step result of imu cov (6,6)
	"""
	Wt = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # (6,6)
		# initialize noise matrix
	mean_next = mean_current @ hp.exp(time*ut4) # 4*4
	cov_next = hp.exp(time*ut6) @ cov_current @ (hp.exp(time*ut6).T) + Wt # 6*6
	return mean_next,cov_next

def part1(twists,time_steps):
	"""
	input the twists obtain from imu measurement, time intervals and the noise matrix W
	output the predicition result: the mean and the covariance of every time steps
	"""
	mean,cov,u4,u6 = imu_prediction_init(twists)
	# initialize noise matrix
	
	for i in range(time_steps.shape[0]):
		mean_next, cov_next = imu_prediction(mean[i,:,:],cov[i,:,:],tau[i],u4[i,:,:],u6[i,:,:])
		mean[i+1,:,:] = mean_next
		cov[i+1,:,:] = cov_next
	return mean,cov



def landmark_init_copy(landmarks, cali_matrix, pose, iTc,bool_matrix):
	"""
	input visual features of every time steps in pixel frame
	output the initialize result and a filter (time step, landmark number) that can filtering out all of the unobserables
	"""
	# landmarks # (4, n, 3026) # [xl,yl,xr,yr]
	

	wTi = np.array([
		[1.,0.,0.,0.],
		[0.,-1.,0.,0.],
		[0.,0.,-1.,0.],
		[0.,0.,0.,1.]
	])

	fsu, fsv, fsub, cu, cv = cali_matrix[0,0], cali_matrix[1,1], -cali_matrix[2,3], cali_matrix[0,2], cali_matrix[1,2]
	ul,vl,ur = landmarks[0,:,:], landmarks[1,:,:], landmarks[2,:,:]

	init_features = np.zeros(landmarks.shape)
	# (4, n, 3026)
	z = init_features[2,bool_matrix] =  fsub / (ul[bool_matrix] - ur[bool_matrix])
	init_features[0,bool_matrix] = np.multiply(z , (ul[bool_matrix] - cu)/fsu)
									# (1, n, 3026) * (1, n, 3026)
	init_features[1,bool_matrix] = np.multiply(z , (vl[bool_matrix] - cv)/fsv)
	init_features[3,bool_matrix] = 1.
	# (4, n, 3026)
	pose = np.tile(pose[np.newaxis,:,:,:],(init_features.shape[1],1,1,1))
	# (n,3024,4,4)
	result =  pose @ wTi @iTc @ np.moveaxis(init_features,0,-1)[:,:,:,np.newaxis] 
	# (n,3024,4,4) @ (n,3024,4)
	return np.moveaxis(np.squeeze(result,-1),-1,0)

def landmark_init(landmarks_i, cali_matrix, pose, iTc,idx_i):
	"""
	input:
		landmarks_i: all the landmarks at time step i in pixel units (4,n)
		cali_matrix: the camera calibration matrix ks
		pose: the tansformation matrix we get from last imu prediction step (4*4)
		idx_i: mask of time step i  ( mask[:,i]: (m,) )
	return:
		the initialization result (4,n)
	"""
	wTi = np.array([
		[1.,0.,0.,0.],
		[0.,-1.,0.,0.],
		[0.,0.,-1.,0.],
		[0.,0.,0.,1.]
	])	# the matrix that help deal with the flip of imu have
	fsu, fsv, fsub, cu, cv = cali_matrix[0,0], cali_matrix[1,1], -cali_matrix[2,3], cali_matrix[0,2], cali_matrix[1,2]
		# getting variables from the calibration matrix Ks
	ul, vl, ur = landmarks_i[0,:], landmarks_i[1,:], landmarks_i[2,:]
		# x_left, y_left, x_right
		# (n,), (n,), (n,)

	init_features = np.zeros(landmarks_i.shape)
		# initialize the initialization result
		# (4, n)
	# plug in all the initialization result for each of the observables and left the unobservables with 0
	z = init_features[2,idx_i] =  fsub / (ul[idx_i] - ur[idx_i])
		# the depth (m,)
	init_features[0,idx_i] = np.multiply(z , (ul[idx_i] - cu)/fsu)
									  # (1, n) * (1, n)
	init_features[1,idx_i] = np.multiply(z , (vl[idx_i] - cv)/fsv)
	init_features[3,idx_i] = 1.
	# (4, n)
					# (4,4) @ (n,4)
	return pose @ wTi @iTc @ init_features

def get_cov_t(cov_total,idx):
	"""
	input:
		cov_total: a marix that store all the covariance at the diagnal form time step 0 to t
					(3n,3n)
		idx: the index of the observable features at both time t and  t+1
	return:
		cov_t: covariance at time i (3m,3m)
		cov_idx: the index used to get the cov_t from cov_total
	"""
	cov_dim = idx.shape[0]*3
	# get the dimension of cov at time i 3m
	dim_idx = np.arange(cov_dim)
	# the index used to put the extracted cov (from cov_total) into a matrix
	cov_idx = np.ravel(np.vstack((3*idx,3*idx+1,3*idx+2)), order='F')
	# find the correspond cov index of the obervables
	cov_t = np.zeros((cov_dim,cov_dim))
	cov_t[dim_idx,dim_idx] = cov_total[cov_idx,cov_idx]
	return cov_t, cov_idx

def ld_H_next(ks,T,miu,N):
	"""
	input:
		ks: calibration matrix: (4,4)
		T: transformation = oTi @ inv(T_t+1) (4,4)
		miu: prior mean in this case we use the initialize result of landmark  (4,m) 
				(m is the overlaping observable features)
	return:
		H: (4m,3m)
	"""
	P = np.hstack((np.eye(3), np.array([0,0,0]).reshape(3,1))) # 3*4
	H = np.zeros((4*N,3*miu.shape[1])) # 4m*3m
	temp = ks @ np.moveaxis(utils.projectionJacobian(np.moveaxis(T@miu,0,-1)),0,-1)
	# 4*4 @ 4*4*m(m*4*4(m*4(4*4 @ 4*m))) = 4*4*m
	h_temp = np.moveaxis(temp,-1,0) @ T @ (P.T)
			# m*4*4 @4*4 @ 4*3 = m*4*3
	for i in range(N):
		H[4*i:4*(i+1),3*i:3*(i+1)] = h_temp[i,:,:]
	return H # 4n*3m

def ld_K_next(cov,H,v):
	V = np.diag(np.full(H.shape[0],v))
	temp1 = cov @ H.T
		# 3m*3m @ 3m*4n = 3m*4n
	temp2 = np.linalg.inv(H @ cov @ H.T + V )
		# 4n*3m @ 3m*3m @ 3m*4n = 4n*4n
	return temp1 @ temp2 # 3m*4n @ 4n*4n = 3m*4n

def landmark_update(Ks,oTi,pose,lm_init,idx,dim,landmark_i,cov_total):
	"""
	input:
		Ks: calibration matrix (4,4)
		oTi: extrinsics (4,4)
		pose: the tansformation matrix we get from last imu prediction step (4,4)
		lm_init: all the landmarks in xyz1 unit after initialization (4,n)
		idx: the index of the obervable at time i (m,)
		landmark_i: all the landmarks at time step i in pixel unit (4,n)
		cov_total: the big covariance matrix (3n,3n)
	return:

	"""
	v = 0.1 # initialize the noise parameter
	T = oTi @ utils.inversePose(pose)
	
	zt = np.ravel(landmark_i[:,idx], order='F') 
		# 4*n -> 4*m ->4m
	cov_t,cov_idx = get_cov_t(cov_total,idx)
		# (3m,3m), (3m,)
	z_predict = Ks @ np.moveaxis(utils.projection(np.moveaxis(T @ lm_init[:,idx],0,-1)),0,-1)
	# 4*4 @ (4*n(n*4(4*n(4*4 @ 4*n)))) = 4*n
	z_predict = np.ravel(z_predict, order='F') # 4n
	H_next = ld_H_next(K_s,T,lm_init[:,idx],dim) # 4n*3m
	K_next = ld_K_next(cov_t,H_next,v)
	# mt = np.ravel(lm_init[:,idx], order='F')# 4*m -> 4m
	# 3*m ->3m
	miu = np.ravel(lm_init[:3,idx], order='F')
	mean_next = miu + K_next @ (zt - z_predict)
		# 4m + 3m*4n @ 4n = 4n
	cov_next = (np.eye(3*dim) -  K_next @ H_next) @ cov_t
		# 3m*3m
	cov_total[cov_idx,cov_idx] = cov_next[np.arange(3*dim),np.arange(3*dim)]
	
	return mean_next, cov_total

def part2(Ks,OTI,landmarks,bool_matrix,mean_pre):
	"""
	input:
		Ks:stereo calibration matrix 
		OTI: cali_matrix  ∈ SE(3)
		landmarks: initialized landmarks in xyz1  (4,n,t)
		bool_matrix 
		mean_pre: new observations zt+1 ∈ R4Nt+1
		cov_pre: 
	return:
		mean: the mean of the 
		cov:
	"""

	timesteps = landmarks.shape[2] 
	num_landmarks = landmarks.shape[1]
	sigma = 0.01
	cov_total = np.diag(np.full(num_landmarks*3, sigma))
	noise = 0.1
	mean_matrix = np.copy(landmarks) # (4,n,t)
	mean_matrix[:3,:,:] = 0
	init = np.copy(mean_matrix)
	for i in range(timesteps - 1):
		idx =  np.argwhere(np.logical_and(bool_matrix[:,i],bool_matrix[:,i+1])).flatten()
		dim = idx.shape[0] # dimension which is m and in this case which is also N_t+1
		lm_init = landmark_init(landmarks[:,:,i],K_s,mean_predict[i,:,:],imu_T_cam,idx)
		init[:,:,i] = lm_init 
		pose = mean_pre[i+1,:,:]
		mean_next, cov_total = landmark_update(Ks,OTI,pose,lm_init,idx,dim,landmarks[:,:,i],cov_total)
		mean_matrix[:3,idx,i+1] = mean_next.reshape((3,dim),order = 'F')
		print(i)
	return mean_matrix,init



def get_z_predict(Ks,oTi,pose,lm_update,idx_i):
	"""
	Ks: calibration matrix (4,4)
	oTi: extrinsics (4,4)
	pose: the tansformation matrix we get from last imu prediction step (4,4)
	lm_update: all the updated landmark at time step i (4,n)
	idx_i: all the index of the both observable at i and i+1
	"""
	prediction = np.zeros((4,lm_update.shape[1])) # (4,n)
	temp = np.moveaxis(oTi @ utils.inversePose(pose) @ lm_update[:,idx_i],-1,0)
		# m*4(4*4 @ 4*4 @ 4*m)
	prediction[:,idx_i] = Ks @ np.moveaxis(utils.projection(temp),-1,0)
		# 4*4 @ 4*m(m*4) = 4*m -> 4*n
	prediction = np.ravel(prediction,order = 'F') # 4n
	return prediction

def imu_H_next(Ks,oTi,pose,lm_update,idx_i):
	"""
	Ks: calibration matrix (4,4)
	oTi: extrinsics (4,4)
	pose: the tansformation matrix we get from last imu prediction step (4,4)
	lm_update: the landmark coords in xyz1 after the prediction step at time step i (4,n)
	idx_i: all the index of the both observable at i and i+1
	"""
	h = np.zeros((lm_update.shape[1],4,6))
		# (n,4,6)
	miu_m = utils.inversePose(pose) @ lm_update[:,idx_i]  
		# (4*4 @ 4*m = 4*m)
	temp1 = utils.projectionJacobian( np.moveaxis((oTi @ miu_m),1,0))
		# m*4*4(n*4(4*4 @ 4*m))
	h[idx_i,:,:] = -Ks @ temp1 @ oTi @ hp.circle_dot(np.moveaxis(miu_m,1,0))
		# 4*4 @ m*4*4 @ m*4*6(m*4) = m*4*6 -> n*4*6
	H = h.reshape(4*h.shape[0],6)  # 4n*6
	H_idx = np.ravel(np.vstack((4*idx_i,4*idx_i+1,4*idx_i+2,4*idx_i+3)), order='F')
		# the index we used to extract the H_next of the obervables from H matrix
	return H, H_idx

def imu_K_next(cov_pre,H_next,H_idx):
	"""
	H_next: (4n*6)
	cov_pre: the covariance matrix from prediction step
	"""
	# initialize noise
	noise = 0.01
	size = H_next[H_idx,:].shape[0] # 4m
	alternating_values = np.array([0.1, 0.5])
	result_vector = np.tile(alternating_values, (size // 2) + 1)[:size]
	V = np.diag(result_vector)

	Kalman_gain = np.empty((6,H_next.shape[0]))
		# 6*6 @ 6*4n(4n*6) @ 4n*4n(4n*6 @ 6*6 @ 6*4n) = 6*4n
	# inv = np.linalg.inv(H_next[H_idx,:] @ cov_pre @ H_next[H_idx,:].T + V)
		# (4m,4m)
	inv = np.linalg.pinv(H_next[H_idx,:] @ cov_pre @ H_next[H_idx,:].T + V)
	Kalman_gain[:,H_idx] = cov_pre @ H_next[H_idx,:].T @ inv
		# (6,4m) -> (6,4n)
	return Kalman_gain

def imu_update(Ks,oTi,pose,lm_update,cov_pre,landmarks_i,idx_i):
	"""
	Ks: calibration matrix (4,4)
	oTi: extrinsics (4,4)
	pose: the tansformation matrix we get from last imu prediction step (4,4)
	lm_update: the updated landmark at time step i (4,n)
	landmarks_i: all the landmarks of time t+1 in pixel unit (4,n)
	idx_i: all the index of the both observable at i and i+1 (m,)

	"""
	z_next = np.ravel(landmarks_i[:,idx_i],order='F')# (4m,)

	z_predict = get_z_predict(Ks,oTi,pose,lm_update,idx_i)# 4*n
	H_next, H_idx = imu_H_next(Ks,oTi,pose,lm_update,idx_i)
	K_next = imu_K_next(cov_pre,H_next,H_idx)	
	mean_next = pose @ hp.exp(utils.axangle2twist(K_next[:,H_idx] @ (z_next - z_predict[H_idx])))
		# 4*4 @ 4*4(6*1(6*4n @ 4n)) = 4*4	
	cov_next = (np.eye(6) - K_next[:,H_idx] @ H_next[H_idx,:]) @ cov_pre
		# (6*6 - (6*4n @  4n*6)) @ 6*6 = 6*6
	return mean_next,cov_next

def SLAM_EKF(time,twists,landmarks,oTi,k_s,bool_matrix):


	# t: all the time steps (3026-1)
	# m: for all the observable landmarks
	# n: for all the landmarks
	steps = time.shape[0] # 3026-1 = 3025 = t
	num_landmarks = landmarks.shape[1] # n 4430
	imu_mean, imu_cov, u4, u6 = imu_prediction_init(twists)
	# (t,4,4) , (t,6,6), (t,4,4) , (t,6,6)

	sigma = 0.01 # initialize a paraments for landmarks cov
	lm_cov_total = np.eye(num_landmarks*3) * sigma # with dimension of (3n,3n)
	lm_mean_total = np.copy(landmarks) # (4,n,t)
	lm_mean_total[:3,:,:] = 0  
	# make sure every landmarks at every time steps is (0,0,0,1)

	for i in range(steps):
			# pre-proccessing
		idx =  np.argwhere(np.logical_and(bool_matrix[:,i],bool_matrix[:,i+1])).flatten() 
			# find the index of the features that are both observable at step i and i+1
			# (m,)
		dimension = idx.shape[0] 
			# dimension: the number of observervables (m) 
		    # m and in this case which is also N_t+1
		lm_mean_current = lm_mean_total[:,:,i]
		imu_mean_current, imu_cov_current = imu_mean[i,:,:],imu_cov[i,:,:] 
			# (4,4), (6,6)
			# getting the imu mean and cov at current time step

			# imu prediction
		imu_mean_pre,imu_cov_pre = imu_prediction(imu_mean_current,imu_cov_current,time[i],u4[i],u6[i])
			# (4,4), (6,6) 
		lm_init = landmark_init(landmarks[:,:,i],k_s,imu_mean_pre,oTi,idx)
		lm_mean_next,lm_cov_total = landmark_update(K_s,oTi,imu_mean_pre,lm_init,idx,dimension,landmarks[:,:,i],lm_cov_total)
		lm_mean_total[:3,idx,i+1] = lm_mean_next.reshape((3,dimension),order = 'F')
		# imu_mean_up, imu_cov_up = imu_update(k_s,oTi,imu_mean_pre,lm_mean_current,imu_cov_pre,landmarks[:,:,i],idx)
		imu_mean_up, imu_cov_up = imu_update(k_s,oTi,imu_mean_pre,lm_mean_total[:,:,i],imu_cov_pre,landmarks[:,:,i],idx)
		
		# post proccessing
		imu_mean[i+1,:,:],imu_cov[i+1,:,:] = imu_mean_up, imu_cov_up
		
		print(i)
	return imu_mean, lm_mean_total


def copy_landmark_update(Ks,OTI,landmarks,z,bool_matrix,mean_pre,cov_pre):
	"""
	input:
		Ks:stereo calibration matrix 
		OTI: cali_matrix  ∈ SE(3)
		landmarks: landmark positions m ∈ R3M,
		z = landmarks in pixel frame
		bool_matrix 
		mean_pre: new observations zt+1 ∈ R4Nt+1
		cov_pre: 
	return:
		mean: the mean of the 
		cov:
	"""
	timesteps = landmarks.shape[2] 
	num_landmarks = landmarks.shape[1]

	V = np.diag([0.1, 0.1, 0.1, 0.1,])
	mean = np.zeros((num_landmarks,timesteps,4,4))
	cov = np.zeros((num_landmarks,timesteps,6,6))
	
	for i in range(timesteps - 1):
		# z_diff.shape (4, 4430)
		# H_next.shape (4430, 4, 6)
		# z_predict.shape (4, 4430)
		# miu_m.shape (4, 4430)
		# temp1.shape (4430, 6, 4)
		# temp2.shape (4430, 4, 4)
		# cov_pre.shape  (3026, 6, 6)
			# find the features that are both observable at step i and i+1
		idx =  np.argwhere(np.logical_and(bool_matrix[:,i], bool_matrix[:,i+1])).flatten()

		miu_m = utils.inversePose(mean_pre[i+1,:,:]) @ landmarks[:,:,i]  # (4*4 @ 4*n = 4*n)
		H_next = -Ks @ utils.projectionJacobian( np.moveaxis((OTI @ miu_m),1,0)) @ OTI @ circle_dot(miu_m)
					# 4*4 @ n*4*4 @ 4*4 @ 4*6 = n*4*6
		temp = np.moveaxis(OTI @ miu_m,0,-1)
				# n*4
		z_predict = Ks @ np.moveaxis(utils.projection(temp),0,-1)
				# 4*4 @ inv.(n*4) = 4*n
		
		temp1 = cov_pre[i+1,:,:] @ np.moveaxis(H_next,1,-1) # 6*6 @ n*6*4
		temp2 = utils.inversePose(H_next @ cov_pre[i+1,:,:] @ np.moveaxis(H_next,1,-1) + V)
					# n*4*6 @ 6*6 @ n*6*4 = n*4*4
		# K_next = torch.matmul(torch.tensor(temp1), torch.tensor(temp2)) # (4430, 6, 4)
		K_next = temp1 @ temp2  # n*6*4
		
		z_diff = np.moveaxis(z[:,:,i+1] - z_predict,0,-1)[:,:,np.newaxis]
				# 4*n -> n*4 -> n*4*1
		# mult1 = np.copy(torch.matmul(K_next, torch.tensor(z_diff)))
		mult1 = K_next @ z_diff
				# n*6*1
		hat = utils.axangle2twist(np.squeeze( mult1,axis =-1 ))
				# n*6*1 -> n*6 -> n*4*4
		exp = np.copy(torch.matrix_exp(torch.tensor(hat)))
		mean[idx,i+1,:,:] = mean_pre[i+1,:,:] @ exp[idx,:,:]
				# n*4*4 = 4*4 @ n*4*4
		# K_next = torch.tensor(K_next)
		# mult2 = np.copy(torch.matmul(K_next, torch.tensor(H_next)))
		cov[idx,i+1,:,:] = (np.eye(6) - K_next[idx,:,:] @ H_next[idx,:,:]) @ cov_pre[i+1,:,:]

		# for j in range(num_landmarks):
		# 	K_next[j,:,:] = temp1[j,:,:] @ temp2[j,:,:]
		# 	hat = utils.axangle2twist(K_next[j,:,:] @ z_diff[:,j])
		# 	mean[i+1,j,:,:] = mean_pre[i+1,:,:] @ sp.linalg.expm(hat)
		# 	cov[i+1,j,:,:] = (np.eye(6) - K_next[j,:,:] @ H_next[j,:,:]) @ cov_pre[i+1,:,:]

		print(i)
	return mean,cov

if __name__ == '__main__':
	# Load the measurements
	filename = "../data/10.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
	"""
					t: (1, 3026)            features: (4, 13289, 3026)
	linear_velocity: (3, 3026)    angular_velocity: (3, 3026)
					K: (3, 3)                      b: ()
			imu_T_cam: (4, 4) camera to imu
	"""

# (a) IMU Localization via EKF Prediction
		# calculate the time interval
	tau = np.diff(t).flatten() # (3025,)
		# put v and w together to get the twist
	twists = np.vstack((linear_velocity,angular_velocity)) # (6, 3026)
		# imu prediction
	mean_predict,cov_predict = part1(twists,tau)
	# (3026, 4, 4)   (3026, 6, 6)

	# 	# visualize trajectory
	plot.p1_plot_traj(mean_predict)
		
# (b) Landmark Mapping via EKF Update
		# initialize the landmarks(from pixel coords to world coords)	
	# features = change_features_form(features,1)
	features = features[:,::3,:]
	# 3*3 -> 4*4
	K_s = hp.get_stereo_ks(K,b)
	mask = hp.find_unobserved(features) # (n, 3026)
		# update step
	# mean_update, init = part2(K_s,imu_T_cam,features,mask,mean_predict)
	# plot.init_plot(mean_predict,init,mask)
	# plot.p2_plot(mean_predict,mean_update,mask)
	
# (c) Visual-Inertial SLAM

	imu_traj, landmarks_pc = SLAM_EKF(tau,twists,features,imu_T_cam,K_s,mask)
	plot.p3_plot(imu_traj,landmarks_pc,mean_predict,mask)
	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


