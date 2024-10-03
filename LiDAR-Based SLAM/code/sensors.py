import numpy as np
import load_data as ld
from sklearn.neighbors import NearestNeighbors
import cv2 as cv
from pr2_utils import bresenham2D
import matplotlib.pyplot as plt

# ld.encoder_counts (4, 4956) 
# ld.encoder_stamps (4956,)
# ld.imu_angular_velocity (3, 12187)
# ld.imu_linear_acceleration (3, 12187)
# ld.imu_stamps (12187,)
# ld.lidar_stamsp.shape (4962,)


def data_sync(b,a):
    """
    aim: synchronize the encoder data and imu data
                    or encoder data and lidar data
    arg:
        
    return:
    """
    indices = np.zeros(b.shape)
    for i in range(b.shape[0]):
        x = np.abs(b[i] - a).argmin()
        indices[i] = x
    return indices

def encoder(z,mpt,tau):
    """
    The encoder sensor model
    arg:
        d: the diameter of the wheel
        z: the encoder count
        mpt: meter per ticks
        tau: the given time duration 
    return:
        the linear velocity of the differencial drive model at every time step
    """
    left_v = (z[0] + z[2])*mpt/2./tau
    right_v = (z[1] + z[3])*mpt/2./tau
    return (right_v + left_v)/2

def mm_differential_drive(state_t,omega,velocity,stamp):
    """
    aim: Build up differential drive motion model
    arg: 

    return: All the state of the differential drive
    """
    return state_t + stamp*np.array([velocity*np.sinc(omega*stamp/2.)*np.cos(state_t[2]+omega*stamp/2.),
                             velocity*np.sinc(omega*stamp/2.)*np.sin(state_t[2]+omega*stamp/2.),
                             omega
                           ])

    # define constants
diameter = 0.254
ticks_per_revolution = 360
meters_per_ticks = 0.0022
counts = 40.

encoder_tau = np.diff(ld.encoder_stamps)

    # synchronize data between encoder and imu
idx = data_sync(ld.encoder_stamps,ld.imu_stamps)
imu_omega = np.zeros((3,idx.shape[0]))
imu_a = np.zeros((3,idx.shape[0]))
imu_stamps = np.zeros((idx.shape[0],))
for x,i in enumerate(idx):
    i = int(i)
    imu_omega[:,x] = ld.imu_angular_velocity[:,i]
    imu_a[:,x] = ld.imu_linear_acceleration[:,i]
    imu_stamps[x] = ld.imu_stamps[i]


    # robot motion model prediction
x = np.array([0.,0.,0.])
v_list = np.array([0.])
states = x # the list of states
for i in range(encoder_tau.shape[0]):
    v = encoder(ld.encoder_counts[:,i],meters_per_ticks,encoder_tau[i])
    v_list = np.append(v_list,v)
    x_next = mm_differential_drive(x,imu_omega[2,i],v,encoder_tau[i])
    states = np.vstack((states,x_next))
    x = x_next
# print(states.shape)  (4956, 3)


    # lidar data scan
    # Each LiDAR scan contains 1081 measured range values
# lidar_ranges   (1081, 4962)     range_max: 30     lidar_range_min:0.1
# angle_max: 2.356194490192345    angle_min: -2.356194490192345   angle_increment  [[0.00436332]]
# stamsp.shape  (4962,)


def copy_get_lidar_coords(ranges,azimuth):
#     """
#     input lidar range return (x,y) coords in lidar frame
#     """
#     elevation = 0. # 2D map
#     multiplier = np.vstack((
#                     np.cos(azimuth)*np.cos(elevation),
#                     np.sin(azimuth)*np.cos(elevation),
#                     np.full(azimuth.shape,np.sin(elevation))
#                 )).T
#     multiplier = np.expand_dims(multiplier, axis=1)
#     multiplier = np.repeat(multiplier,ranges.shape[1], axis=1)
#     temp = np.expand_dims(ranges, axis=2)
#     temp = np.repeat(temp, 3, axis=2)
     return #temp*multiplier

#    # changing the ranges to lidar coords
# lidar_coords = get_lidar_coords(lidar_ranges,lidar_angles)  # (1081, 4962, 3)

def get_lidar_coords(ranges,azimuth):
    """
    input lidar range return (x,y) coords in lidar frame
    """
    elevation = 0. # 2D map
    multiplier = np.vstack((
                    np.cos(azimuth)*np.cos(elevation),
                    np.sin(azimuth)*np.cos(elevation),
                    np.full(azimuth.shape,np.sin(elevation))
                )).T
    temp = np.tile(ranges, (3, 1)).T
    return temp*multiplier

    # synchronize data between encoder and lidar
idx_1 = data_sync(ld.encoder_stamps,ld.lidar_stamsp)
lidar_ranges = np.zeros((ld.lidar_ranges.shape[0],idx.shape[0]))
    # getting the  lidar ranges in the right time steps  (1081, 4956)
for x,i in enumerate(idx_1):
    i = int(i)
    lidar_ranges[:,x] = ld.lidar_ranges[:,i]

    # getting the angles for all the lidar
lidar_angles = np.zeros(lidar_ranges.shape[0])
increment = ld.lidar_angle_increment[0,0]
for i in range(lidar_angles.shape[0]):
    lidar_angles[i] = ld.lidar_angle_min + i*increment

# print(lidar_angles.shape) # (1081,)
# print(lidar_ranges.shape) # (1081, 4956)

def sensor2robot(sensor_coords):
    """
    input a (x,y) coords in sensor frame 
    return a (x,y) coords in robot frame
    """
    T = np.eye(4)
    T[:3,3] = np.array([ 300. - 330.20/2., 0., 514.35])
    # homo_form= np.concatenate((sensor_coords, 
    #                             np.ones((sensor_coords.shape[0],sensor_coords.shape[1], 1))), 
    #                             axis=2)
    # expended_homo_form = np.expand_dims(homo_form,axis = 3)

    # expended_T = np.expand_dims(T,axis = (0,1))
    # expended_T = np.repeat(expended_T,homo_form.shape[0], axis=0)
    # expended_T = np.repeat(expended_T,homo_form.shape[1], axis=1)

    # result = np.matmul(expended_T,expended_homo_form)
    # return np.squeeze(result,axis=3)
    homo_form = np.hstack((sensor_coords,np.ones((sensor_coords.shape[0],1)))) 
    return np.dot(homo_form,T)
    
    
# changing the points from lidar frame to robot frame

#robot_coords = sensor2robot(lidar_coords)  # (1081, 4956, 4)
# print(robot_coords.shape)



# changing the state (x,y,theta) into transformation matrices
def state2transform(state_list):
    T_list = np.tile(np.eye(4),(state_list.shape[0],1,1))
    for i,x in enumerate(state_list):
        T_list[i,:3,:3] = np.array([
            [np.cos(x[2]),-np.sin(x[2]),0.],
            [np.sin(x[2]),np.cos(x[2]),0.],
            [0.,0.,1.]
        ])

        T_list[i,:3,3] = np.array([x[0],x[1],0.])
    return T_list
T_list = state2transform(states)  # (4956, 4, 4)
# print(T_list.shape)

# robot_coords = robot_coords.reshape(robot_coords.shape[1],robot_coords.shape[1],robot_coords.shape[2])

def robot2world(point_cloud,transformation):
    """
    input robot frame coords, and Transformation matracies at every time step
    output coords in world frame (x,y,z,1)
    """
    # result = np.zeros(point_cloud.shape)
    # for i,t in enumerate(matrices):
    #     for x in range(point_cloud.shape[0]):
    #         result[x,i,:] = t@point_cloud[x,i,:]
    
    # return result
    return np.dot(point_cloud,transformation)
# pc_world = robot2world(robot_coords,T_list) # (1081, 4956, 4)
# print(pc_world.shape)
# print(pc_world[:,:,0])

    # occupancy mapping
def get_occupancy_map(point_cloud,state,canvas):

    for j,x in enumerate(point_cloud):

        sx = np.ceil((state[0] - canvas['xmin']) / canvas['res'] ).astype(np.int16)-1
        sy = np.ceil((state[1] - canvas['ymin']) / canvas['res'] ).astype(np.int16)-1
        ex = np.ceil((x[0] - canvas['xmin']) / canvas['res'] ).astype(np.int16)-1
        ey = np.ceil((x[1] - canvas['ymin']) / canvas['res'] ).astype(np.int16)-1
        points = bresenham2D(sx,sy,ex,ey)
            # convert from meters to cells

        # build an arbitrary map 
        indGood = np.logical_and(np.logical_and(np.logical_and((points[0] > 1), (points[1] > 1)), (points[0] < canvas['sizex'])), (points[1] < canvas['sizey']))
        
        point_x = points[0, indGood].astype(int)
        point_y = points[1, indGood].astype(int)
        canvas['map'][point_x, point_y] = 1
    return canvas




# build occupancy map
    # init MAP
MAP = {}
MAP['res']   = 0.05 #meters
MAP['xmin']  = -30  #meters
MAP['ymin']  = -30
MAP['xmax']  =  30
MAP['ymax']  =  30
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
# (1201, 1201)
MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8




for i in range(lidar_ranges.shape[1]): # doing this for every time steps
        # take valid indices
    indValid = np.logical_and((lidar_ranges[:,i] < 30),(lidar_ranges[:,i]> 0.1))
    ranges = lidar_ranges[:,i][indValid]
    angles = lidar_angles[indValid]

    lidar_coords = get_lidar_coords(ranges,angles) # (n,3) 3 for x,y,0
    robot_coords = sensor2robot(lidar_coords) # (n,4) for x,y,z,1
    world_coords = robot2world(robot_coords,T_list[i])  # (n,4) for x,y,z,1
    MAP = get_occupancy_map(world_coords,states[i],MAP)
      

    if i%300 == 0:

        zero_count = np.count_nonzero(MAP['map'] == 0)
        ones_count = np.count_nonzero(MAP['map'] == 1)
        print("there's ",zero_count," zeros")
        print("there's ",ones_count," ones")
        fig2 = plt.figure()
        plt.imshow(MAP['map'],cmap="hot")
        plt.title("Occupancy grid map")
        plt.grid(True)
        plt.show()











