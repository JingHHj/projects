import numpy as np
import pr3_utils as utils
import torch






x = np.array([
    [1,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,8,9]
])
y = np.ravel(x, order='F')
# print(y)
# print(y.reshape((4,9),order='F'))

a = np.array([
	[1,2,3,4],
	[1,2,3,4],
	[1,2,3,4]
])
b = np.array([2,2,2,2])

print(a @ b)
print(np.matmul(a,b))


def landmark_update(Ks,OTI,landmarks,z,bool_matrix,mean_pre,cov_pre):
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
	P = np.hstack((np.eye(3), np.array([0,0,0]).reshape(3,1)))
	sigma = np.diag([ 0.1, 0.1, 0.1,])
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

			# 4*4 @ 4*4 = 4*4
		T = OTI @ utils.inversePose(mean_pre[i+1,:,:])
			# (4*4 @ 4*n = 4*n -> n*4) -> 4*4 (n*4 ->4*n) = 4*n
		# landmark_flatten = np.ravel(landmarks[:,:,i], order= 'F') # 4n
		ld = landmarks[:,bool_matrix[:,i],i] # 4*n
		z_predict = Ks @ np.moveaxis(utils.projection(np.moveaxis(T @ ld,0,-1)),0,-1)
			# 4*n =	4*4 @ 4*n <- (n*4 = (n*4 = (4*4 @ 4*n = 4*n) ) ) )

				# (4*4 @ 4*n = 4*n) = n*4) =n*4*4       
		temp1 = utils.projectionJacobian(T @ np.moveaxis(ld,0,-1))
				# 4*4 @ 4*3 = 4*3 -> n*4*3
		temp2 = np.tile(T @ P.T,(ld.shape[1],1,1) )
				# 4*4 @ n*4*4 @ n*4*3 = n*4*3
		H_next = Ks @ temp1 @ temp2

		# K_next = 
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