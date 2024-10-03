import numpy as np
from pr3_utils import *
import pr3_utils as utils
import scipy as sp
import plot
import torch

def imu_prediction(mean_current,cov_current,time,ut4,ut6,Wt):

	mean_next = mean_current @ sp.linalg.expm(time*ut4)
	exp = np.copy(torch.matrix_exp(torch.tensor(time*ut6)))
	cov_next = exp @ cov_current @ (sp.linalg.expm(time*ut6).T) + Wt
	return mean_next,cov_next

def part1(twists,noise_matrix,time_steps):
	"""
	input the twists obtain from imu measurement, time intervals and the noise matrix W
	output the predicition result: the mean and the covariance of every time steps
	"""
	if twists.shape[0] == 6:
		twists = twists.T
		
	u4 = utils.axangle2twist(twists) # (3026, 4, 4)
	u6 = utils.axangle2adtwist(twists) # (3026, 6, 6)
	# pose zero
	mean = np.tile(np.eye(4),(u4.shape[0],1,1)) # 3026*4*4
	cov_0 = np.diag([0.1,0.1,0.1,0.1,0.1,0.1])
	cov = np.tile(cov_0,(u4.shape[0],1,1)) # 3026*6*6
	
	for i in range(time_steps.shape[0]):
		mean_next, cov_next = imu_prediction(mean[i,:,:],cov[i,:,:],tau[i],u4[i,:,:],u6[i,:,:],noise_matrix)
		mean[i+1,:,:] = mean_next
		cov[i+1,:,:] = cov_next
	return mean,cov


def landmark_prediction(pose,oTi,landmarks_i,filter_i,filter_next,ob,cov_total,mean_total,Ks):
	
		T = oTi @ utils.inversePose(pose)
			# find the index of the features that are both observable at step i and i+1
		idx =  np.argwhere(np.logical_and(filter_i, filter_next)).flatten()
		dim = idx.shape[0] # dimension which is m and in this case which is also N_t+1
		mt =  landmarks_i[:,idx] # 4*m
		
		zt = np.ravel(ob[:,idx], order='F') # 4n in this case n =m
		cov_t,cov_idx = get_cov_t(cov_total,idx)
		z_predict = Ks @ np.moveaxis(utils.projection(np.moveaxis(T @ mt,0,-1)),0,-1)
		# 4*4 @ (4*n(n*4(4*n(4*4 @ 4*n)))) = 4*n
		z_predict = np.ravel(z_predict, order='F') # 4n
		N = int(zt.shape[0]/4)
		H_next = get_H_next(K_s,T,mt,N) # 4n*3m
		K_next = get_K_next(cov_t,H_next,noise)
		mt = np.ravel(mt, order='F')

		# 4*m -> 4m
		# 3*m
		miu = np.ravel(landmarks_i[:3,idx], order='F')
		mean_next = miu + K_next @ (zt - z_predict)
			# 4m + 3m*4n @ 4n = 4n
		cov_next = (np.eye(3*dim) -  K_next @ H_next) @ cov_t
			# 3m*3m
		cov_total[cov_idx,cov_idx] = cov_next[np.arange(3*dim),np.arange(3*dim)]

		return mean_next, cov_next

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

def landmark_init(landmarks, cali_matrix, pose, iTc):
	"""
	input visual features of every time steps in pixel frame
	output the initialize result and a filter (time step, landmark number) that can filtering out all of the unobserables
	"""
	# landmarks # (4, n, 3026) # [xl,yl,xr,yr]
	bool_matrix = find_unobserved(landmarks) # (n, 3026)

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
	return np.moveaxis(np.squeeze(result,-1),-1,0), bool_matrix

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

def get_cov_t(cov_total,idx):
	dim = idx.shape[0]*3
	dim_idx = np.arange(dim)
	cov_idx = np.ravel(np.vstack((3*idx,3*idx+1,3*idx+2)), order='F')
	cov_t = np.zeros((dim,dim))
	cov_t[dim_idx,dim_idx] = cov_total[cov_idx,cov_idx]
	# for i in range(dim):
	# 	cov_t[3*i:3(i+1),3*i:3(i+1)]
	return cov_t, cov_idx

def get_H_next(ks,T,miu,N):
	"""
	ks: calibration matrix: 4*4
	T: transformation = oTi @ inv(T_t+1) 4*4
	miu: prior mean in this case we use the initialize result of landmark 4*m (m is the overlaping observable features)

	H: 4n*3m
	"""
	P = np.hstack((np.eye(3), np.array([0,0,0]).reshape(3,1)))
	H = np.zeros((4*N,3*miu.shape[1])) 
	temp = ks @ np.moveaxis(utils.projectionJacobian(np.moveaxis(T@miu,0,-1)),0,-1)
	# 4*4 @ 4*4*m(m*4*4(m*4(4*4 @ 4*m))) = 4*4*m
	h_temp = np.moveaxis(temp,-1,0) @ T @ (P.T)
			# m*4*4 @4*4 @ 4*3 = m*4*3
	for i in range(N):
		H[4*i:4*(i+1),3*i:3*(i+1)] = h_temp[i,:,:]
	return H # 4n*3m

def get_K_next(cov,H,v):
	V = np.diag(np.full(H.shape[0],v))
	temp1 = cov @ H.T
		# 3m*3m @ 3m*4n = 3m*4n
	temp2 = utils.inversePose(H @ cov @ H.T + V )
		# 4n*3m @ 3m*3m @ 3m*4n = 4n*4n
	return temp1 @ temp2 # 3m*4n @ 4n*4n = 3m*4n



def part2(Ks,OTI,landmarks,observation,bool_matrix,mean_pre):
	"""
	input:
		Ks:stereo calibration matrix 
		OTI: cali_matrix  ∈ SE(3)
		landmarks: landmark positions m ∈ R3M,
		observation = landmarks in pixel frame (4,n,t)
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
	covariance = np.diag(np.full(num_landmarks*3, sigma))
	noise = 0.1
	mean_matrix = np.copy(landmarks)
	for i in range(timesteps - 1):
		
		# mean_next, cov_next = landmark_prediction(mean_pre[i+1,:,:],OTI,landmarks[:,:,i],bool_matrix[:,i],bool_matrix[:,i+1],observation[:,:,i])

		T = OTI @ utils.inversePose(mean_pre[i+1,:,:])
			# find the index of the features that are both observable at step i and i+1
		idx =  np.argwhere(np.logical_and(bool_matrix[:,i], bool_matrix[:,i+1])).flatten()
		dim = idx.shape[0] # dimension which is m and in this case which is also N_t+1
		mt =  landmarks[:,idx,i] # 4*m
		
		zt = np.ravel(observation[:,idx,i], order='F') # 4n in this case n =m
		cov_t,cov_idx = get_cov_t(covariance,idx)
		z_predict = Ks @ np.moveaxis(utils.projection(np.moveaxis(T @ mt,0,-1)),0,-1)
		# 4*4 @ (4*n(n*4(4*n(4*4 @ 4*n)))) = 4*n
		z_predict = np.ravel(z_predict, order='F') # 4n
		N = int(zt.shape[0]/4)
		H_next = get_H_next(K_s,T,mt,N) # 4n*3m
		K_next = get_K_next(cov_t,H_next,noise)
		mt = np.ravel(mt, order='F')

		# 4*m -> 4m
		# 3*m
		miu = np.ravel(landmarks[:3,idx,i], order='F')
		mean_next = miu + K_next @ (zt - z_predict)
			# 4m + 3m*4n @ 4n = 4n
		cov_next = (np.eye(3*dim) -  K_next @ H_next) @ cov_t
			# 3m*3m
		covariance[cov_idx,cov_idx] = cov_next[np.arange(3*dim),np.arange(3*dim)]
		mean_matrix[:3,idx,i+1] = mean_next.reshape((3,dim),order = 'F')
		print(i)
	return mean_matrix,covariance

def SLAM_predict(mean_tt,cov_tt,ut4,ut6,tau,W):
	"""
	mean_tt: pose at t (4,4)
	"""
	
	mean_tnext = mean_tt @ sp.linalg.expm(tau*ut4)
	exp = np.copy(torch.matrix_exp(torch.tensor(tau*ut6)))
	cov_tnext = exp @ cov_tt @ (sp.linalg.expm(tau*ut6).T) + W
	return mean_tnext,cov_tnext

def SLAM_update(mean_pre,cov_pre):
	mean_next_next, cov_next_next = 0,0
	return mean_next_next, cov_next_next

def SLAM_EKF(time,twists,landmarks,bool_matrix,oTi,k_s):
	timesteps = landmarks.shape[2]
	num_landmarks = landmarks.shape[1]
	
	if twists.shape[0] == 6:
		twists = twists.T
	u4 = utils.axangle2twist(twists) # (3026, 4, 4)
	u6 = utils.axangle2adtwist(twists) # (3026, 6, 6) 
	mean_total = np.tile(np.eye(4),(4,4,num_landmarks,timesteps))
	cov_total = np.tile(np.eye(6),(6,6,num_landmarks,timesteps))
	mean_init = np.zeros((4,4,num_landmarks))
	cov_init = np.zeros((6,6,num_landmarks))

	for i in range(timesteps):
		idx =  np.argwhere(np.logical_and(bool_matrix[:,i], bool_matrix[:,i+1])).flatten()
		mean_init[4,4,idx] = mean_total[:,:,idx,i]
		cov_init[6,6,idx] = cov_total[:,:,idx,i]
		mean_predict, cov_predict = imu_prediction(mean_current,cov_current,u4[i],u6[i],time[i])
		mean_update, cov_update = SLAM_update()
		mean_total

		mean_current = mean_update
		cov_current = cov_update



	return


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
		# initialize noise matrix
	noise = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
		# imu prediction
	mean_predict,cov_predict = part1(twists,noise,tau)
	# (3026, 4, 4)   (3026, 6, 6)
	# 	# visualize trajectory
	plot.p1_plot_traj(mean_predict)
		
# (b) Landmark Mapping via EKF Update
		# initialize the landmarks(from pixel coords to world coords)	
	# features = change_features_form(features,1)
	K_s = get_stereo_ks(K,b)
		# (3, n, 3026)
	init_features, mask = landmark_init(features,K_s,mean_predict,imu_T_cam)
		# (4, n, 3026)  (n, 3026)
	plot.init_plot(mean_predict,init_features,mask)
	# 3*3 -> 4*4
		# update step
	mean_update, cov_update = part2(K_s,imu_T_cam,init_features,features,mask,mean_predict)
	plot.p2_plot(mean_predict,mean_update,init_features,mask)
	
# (c) Visual-Inertial SLAM


	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)


