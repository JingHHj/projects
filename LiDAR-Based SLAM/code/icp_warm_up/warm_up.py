import numpy as np
import utils 
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


model_name = 'drill'
id = 1
# read point cloud
pc = utils.load_pc(model_name,id)  # (11690, 3)
# read canonical model
canonical_model = utils.read_canonical_model(model_name) # (14298, 3)  mi


def data_association(source,target):
        # doing nearest Neighbors on the set
   
    # initialize NearestNeighbors ，set K=1（find the nearest neighor），algorithm='ball_tree'
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(source)
    # use kneighbors() find the nearest neighbor in target
    distances, indices = nbrs.kneighbors(target)
    return source[indices].reshape(-1,3), distances  # zi

    # doing Kabsch Algorithm find R
def Kabsch_Alg(source,target):
    """
    input
    """ 
    # find point cloud centroids     
    mean_z = np.mean(source,axis = 0)
    mean_m = np.mean(target,axis = 0)
    # centered point clouds
    delta_z = source - mean_z
    delta_m = target - mean_m
    # find Q matrix
    Q = np.zeros((3,3))
    for i in range(delta_z.shape[0]):
        mult = np.outer(delta_m[i],delta_z[i])
        Q = Q + mult

    # doing svd decomposition to matrix Q
    U, S, Vt = np.linalg.svd(Q)

    # composing rotation matrix R
    M = np.array([
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,np.linalg.det(U@Vt)]
    ])
    R = U@M@Vt
    # find translation P
    # p = mean_m - R.T@mean_z  # best

    # p = R.T@mean_z - mean_m   # best
    # p = mean_z - R@mean_m 

    # p = mean_m - R@mean_z  # given
    # p = mean_z -R.T@mean_m
    # construct se(3)
    # return np.vstack((
    #                 np.hstack((R,p.reshape(3,1))),
    #                 np.array([0.,0.,0.,1.])
    #             ))
    return R


def icp(steps,z,m):
    """
    """
    # initialization

    # transformation = np.array([
    #                 [np.cos(np.pi/3.),-np.sin(np.pi/3.), 0.,0.],
    #                 [np.sin(np.pi/3.), np.cos(np.pi/3.), 0.,0.],
    #                 [0.,               0.,               1.,0.],
    #                 [0.,0.,0.,1.]
    #                         ])
    # t = transformation

    R = np.array([
                    [np.cos(np.pi/3.),-np.sin(np.pi/3.), 0.],
                    [np.sin(np.pi/3.), np.cos(np.pi/3.), 0.],
                    [0.,               0.,               1.]
                            ])
    p = np.array([0.,0.,0.])
    
    # p_0 = np.mean(z,axis=0,keepdims=True) - np.mean(m,axis=0,keepdims=True)
    # t[:3,3] = p_0
    m_init = np.copy(m)
    # utils.visualize_icp_result(z,m_init,t)
    
    errors = []
    for i in range(steps):
        # initialization
        # for j in range(m.shape[0]):
            # m_init[j] = t[:3,:3]@(m_init[j] - t[:3,3])    # BEST
            # m_init[j] = t[:3,:3]@m_init[j] + t[:3,3]   # given
            # m_init[j] = t[:3,:3]@(m_init[j] + t[:3,3]) 
            
            # m_init[j] = t[:3,:3].T@(m_init[j] - t[:3,3])  

            # m_init[j] = t[:3,:3]@m_init[j] - t[:3,3]
            # m_init[j] = t[:3,:3].T@m_init[j] - t[:3,3]
            
        m_init = np.dot(m_init,R) + p

        z_associated,distances = data_association(z,m_init)

        rotation = Kabsch_Alg(z_associated,m_init)
        p_1 = np.mean(z_associated,axis = 0) - np.mean(np.dot(m_init,R),axis = 0)
        R = rotation@R
 
        p = rotation@p_1 + p
        
        # error
        mean_error = np.mean(p_1)
        errors.append(mean_error)  
        # print("iteration: ",i," ,error: ",mean_error)

        t = np.vstack((
            np.hstack((rotation,p_1.reshape(3,1))),
            np.array([0.,0.,0.,1.])
        ))
        
    utils.visualize_icp_result(z,m_init,t)

    
    return t, errors


# thetalist = np.arange(-np.pi/12,np.pi/12,np.pi/60)
# for x in thetalist:
#     t = icp(60,canonical_model,pc,x)
#     print(x)

steps = 100
T,errors = icp(steps,canonical_model,pc)
# utils.visualize_icp_result(canonical_model,pc,T)


    # plotting errors
errors = errors[50:]
plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-')
plt.xlabel('Iteration')
plt.ylabel('Mean Error')
plt.title('ICP Mean Error over Iterations')
plt.grid(True)
plt.show()

