import numpy as np
from sklearn.neighbors import NearestNeighbors

def Kabsch_Algorithm(source, target):
    source = source - np.mean(source,axis=0)
    target = target - np.mean(target,axis=0)
    Q_matrix = np.zeros((3,3))
    for i in range(source.shape[0]):
        Q_matrix = Q_matrix + np.outer(source[i], target[i].T)
        # pdb.set_trace()
    U, S, V_T = np.linalg.svd(Q_matrix)
    # avoid reflection
    F = np.array([[1, 0, 0], 
                  [0, 1, 0], 
                  [0, 0, np.linalg.det(U @ V_T)]])
    R_optimal = U @ F @ V_T
    return R_optimal

def icp(source, target, down_sample_rate, object, data_num,max_iterations=150, tolerance=1e-5):
    '''
    Iterative Closest Point (ICP) algorithm
    source: numpy array, (N, 3)
    target: numpy array, (N, 3)
    max_iterations: int, maximum number of iterations
    tolerance: float, difference in error to stop iterations
    return: SO(3) numpy array, (3, 3), Rotation matrix
    '''
    # difference in center of mass
    p_0 = np.mean(source,axis=0,keepdims=True) - np.mean(target,axis=0,keepdims=True)
    if down_sample_rate == 1:
        source_pc_downsampled = source
    else:
        source_pc_downsampled = source[::int(down_sample_rate/5)]
    target_original = target.copy()
    source_original = source.copy()
    target = target + p_0
    target_pc_downsampled = target[::down_sample_rate]
    guessed_angel = initialize_rotation(source_pc_downsampled,target_pc_downsampled,object)
    Rot = Rotation.from_euler('z',guessed_angel).as_matrix()
    Old_Rot = Rot
    Old_Trans = p_0 
    
    rot_target_pc_downsampled =  target_pc_downsampled @ Rot.T
    for i in range(max_iterations):
        # Find the nearest neighbors
        associated_source, target_pc = data_association(source_pc_downsampled, rot_target_pc_downsampled)
        R = Kabsch_Algorithm(associated_source, target_pc)
        print("ICP iteration: ",i)
        LOSS = loss(associated_source, target_pc, R)
        print("Loss: ",LOSS)
        # if LOSS < terminated_loss[f'{data_num}']:
        #     break
        rot_target_pc_downsampled =  target_pc @ R.T
        translation = np.mean(associated_source,axis=0) - np.mean(rot_target_pc_downsampled,axis=0)
        New_Rot = R @ Old_Rot
        # pdb.set_trace()
        # New_Trans = (R @ Old_Trans.T).T + translation
        New_Trans = Old_Trans @ R.T + translation
        Old_Rot = New_Rot
        Old_Trans = New_Trans
        # print("new translation",New_Trans)
        rot_target_pc_downsampled = rot_target_pc_downsampled + translation

    Optimal_translation = np.mean(rot_target_pc_downsampled,axis=0,keepdims=True) - np.mean(target_original,axis=0,keepdims=True) @ New_Rot.T
    Optimal_translation_inverse = np.mean(target_original,axis=0,keepdims=True) - np.mean(rot_target_pc_downsampled,axis=0,keepdims=True) @ New_Rot
    T_target_to_source = homegenous_transformation(New_Rot, Optimal_translation)
    T_source_to_target = homegenous_transformation(New_Rot.T, Optimal_translation_inverse)
    return rot_target_pc_downsampled, T_source_to_target

def data_association(source_pc, target_pc):
    # print("Numbers of Neighbors",target_pc.shape[0])
    neighbbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(source_pc)
    distances, indices = neighbbors.kneighbors(target_pc)
    assoc_source = source_pc[indices].reshape(-1,3)
    return assoc_source, target_pc
