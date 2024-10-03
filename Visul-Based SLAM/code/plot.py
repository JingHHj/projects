import numpy as np
import matplotlib.pyplot as plt

def p1_plot_traj(T_list):
    # get x,y
    x = T_list[:, 0, 3]
    y = T_list[:, 1, 3]
    # create plot
    plt.plot(x, y)
    # title and label
    plt.title('imu trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.grid(True)
    # show plot
    plt.show()

    return None


def p2_plot(T_list,point_cloud,mask):

    x1,y1 = T_list[:,0,3], T_list[:,1,3]
    x2,y2 = point_cloud[0,mask], point_cloud[1,mask]
    # x3,y3  = init[0,mask], init[1,mask]
    indices = np.argwhere((np.abs(x2) < 1500) & (np.abs(y2) < 1500)).flatten()
    plt.plot(x1, y1, label='imu prediction')
    # plt.scatter(x3[indices], y3[indices], marker='.', color='green',label='initialization', alpha=0.8)
    plt.scatter(x2[indices], y2[indices], marker='.', color='red',label='update step', alpha=0.8)
    
    # title
    plt.title('update step')
    plt.xlim(-1500, 1500)
    plt.ylim(-1500, 1500)       
    plt.grid(True)
    plt.show()

    return None


def init_plot(T_list,point_cloud,mask):

    x1 = T_list[:,0,3]
    y1 = T_list[:,1,3]
    x2 = point_cloud[0,mask]
    y2 = point_cloud[1,mask]
    indices = np.argwhere((np.abs(x2) < 1250) & (np.abs(y2) < 1250)).flatten()
    plt.plot(x1, y1, label='imu trajectory')
    plt.scatter(x2[indices], y2[indices], marker='.', color='red',label='landmarks', alpha=0.5)
    plt.legend()
    plt.title('the initialization result of part 2')
    
    plt.xlim(-1500, 1500)
    plt.ylim(-1500, 1500)
    plt.grid(True)
    plt.show()

    return None

def p3_plot(T_list,point_cloud,p1,mask):

    x1,y1 = T_list[:,0,3], T_list[:,1,3]
    x2,y2 = point_cloud[0,mask], point_cloud[1,mask]
    x3,y3  = p1[:,0,3], p1[:,1,3]
    indices = np.argwhere((np.abs(x2) < 1500) & (np.abs(y2) < 1500)).flatten()
    plt.plot(x1, y1, label='p3', color='green')
    plt.plot(x3, y3, label='p1', color='blue')
    # plt.scatter(x3[indices], y3[indices], marker='.', color='green',label='initialization', alpha=0.8)
    plt.scatter(x2[indices], y2[indices], marker='.', color='red',label='landmarks', alpha=0.8)
    
    plt.legend()
    plt.title('part3')

    plt.xlim(-1500, 1500)
    plt.ylim(-1500, 1500)       
    plt.grid(True)
    plt.show()

    return None