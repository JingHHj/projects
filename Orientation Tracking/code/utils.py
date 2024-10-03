import jax.numpy as jnp
import jax
import numpy as np
import transforms3d as tf3d 
from load_data import imud

from load_data import vicd

# tau_imud shape: (5644, 1)
# raw_imud shape: (5645, 6)


"""helper functions: """
def quaternion_mult(q,p):
    """
    aim: computing quternion multiplication

    arg:
    q: a quaternion ( a 1*4 vector )
    p: a quaternion ( a 1*4 vector )

    return: the product of q and p , which is a quaternion ( a 1*4 vector )
    """
    real = q[0]*p[0] - jnp.transpose(q[1:])@p[1:] 
        # compute real part of the quaternion
    im = q[0]*p[1:] + p[0]*q[1:] + jnp.cross(q[1:],p[1:])
        # compute imaginary part of the quaternion
    return jnp.array([real,im[0],im[1],im[2]])
        # return the quaternion

def quaternion_norm(q):
    """
    aim:compute the norm of a quaternion

    arg:
    q: a quaternion ( a 1*4 vector )

    return: the norm of the input quternion, ( a number )

    """

    norm = (q[0]**2+q[1:]@jnp.transpose(q[1:]))**0.5
    return norm + 0.000001

def quaternion_inv(q):
    """
    aim: compute the inverse of a quaternion

    arg:
    q: a quaternion ( a 1*4 vector )

    return: the inverse of the input quaternion ( a 1*4 vector )

    """
    sqrt_norm = quaternion_norm(q)**2
        # computing the square of the norm
    return jnp.array([q[0]/sqrt_norm,-q[1]/sqrt_norm,-q[2]/sqrt_norm,-q[3]/sqrt_norm])
        # computing and returning the inverse

def quaternion_exp(q):
    """
    aim: computing the quternion's exponential

    arg:
    q: a quaternion ( a 1*4 vector )

    return: the exponential of the input quaternion ( a 1*4 vector )
    """
    real_exp = jnp.exp(q[0])
        # compute the exponential of the real part, which is a number
    qv_norm = (jnp.transpose(q[1:])@q[1:])**0.5
        # compute the norm of the imaginary part
    return jnp.array([real_exp*jnp.cos(qv_norm),
                      real_exp*jnp.sin(qv_norm)*q[1]/qv_norm,
                      real_exp*jnp.sin(qv_norm)*q[2]/qv_norm,
                      real_exp*jnp.sin(qv_norm)*q[3]/qv_norm])

def quaternion_log(q):
    """
    aim: to compute the logarithm of a quaternion

    arg:
    q: a quaternion( a 1*4 vector )

    return: the logarithm
    """
    q = q + 0.0000001
    real_log = jnp.log(q[0])
        # compute the logarithm of the real part, which is a number
    qv_norm = (jnp.transpose(q[1:])@q[1:])**0.5
        # compute the norm of the imaginary part
    return jnp.array([real_log,
                      (q[1]/qv_norm)*jnp.arccos(q[0]/quaternion_norm(q)),
                      (q[2]/qv_norm)*jnp.arccos(q[0]/quaternion_norm(q)),
                      (q[3]/qv_norm)*jnp.arccos(q[0]/quaternion_norm(q))])

"""calibration: """
def get_biases(imu_data):
    """
    aim: compute the basis for the accelorater and angular velocity
    arg: 
        imud: the IMU data 
    return: a vetor of the baises for each variable (ax,ay,az,wx.wy,wz) among the first 50 step

    """
    return jnp.mean(imu_data[0:101,:],axis=0)

def get_calibrated(raw_input,input_biases):
    """
    aim:
    arg:
    return:
    """
    scale_factor_a = jnp.full(raw_input[:,:3].shape, 3300./1023./300.)
    scale_factor_w = jnp.full(raw_input[:,3:].shape, 3300./1023./3.33*(jnp.pi/180.))
    biases = jnp.tile(input_biases, (raw_input.shape[0], 1))
    scale_factor = jnp.hstack((scale_factor_a,scale_factor_w))
    new = (raw_input - biases)*scale_factor
    return new

def motion_model(tau,omega,q_t):  
    """
    aim: computing the qutarnion of next time step by using the quternion exponential

    arg:
    tau_t: the time interval between this time step and next time step (a number)
    omega_t: the angular velocity at this time step ( a 1*3 vector )
    q_t: the qutaternion at this time step ( a 1*4 vector )

    return: the quaternion at next time step
    """ 

    q_exp = jnp.hstack([0.,omega*tau/2.])
    return quaternion_mult(q_t, quaternion_exp(q_exp))

def flip_ax_ay(data):
    """
    aim: to flip the information of acceloration on x and y axis
    arg:
        data: calibrated imu data

    return: flipped data
    """
    temp_ax = -data[:,0]
    temp_ay = -data[:,1]
    temp_az = data[:,2]
    temp_w = data[:,3:6]
    a = jnp.vstack((temp_ax,temp_ay,temp_az))
    return jnp.hstack((a.transpose(),temp_w))




    # motion model

def mm_initial(res, el):
    """
    - `res`: The result from the previous loop.
    - `el`: The current array element.
    """

    """
    aim:
    arg:
    return:
    """
    tau = el[6]
    omega = el[3:6]
    result = motion_model(tau,omega,res)
    return result,result

def observation_model(q_t):
    """

    aim: estimating the acceloration with garvity and the quaternion
    arg:
        q_t: the quaternion at current time step
    
    return: the estimated acceloration, a quaternion (1*4 ndarray)
    """
    g_qua = jnp.array([0.,0.,0.,1.])
    return quaternion_mult(quaternion_inv(q_t),
                    quaternion_mult(g_qua,q_t))

def loss_func( q, q_next, imu_a_qua, imu_w, imu_tau):
    """
    aim: the loss function
    arg:

    return:
    """
    
    part_1 = quaternion_norm(
                    2.*quaternion_log(
                        quaternion_mult(
                            quaternion_inv(q_next),motion_model(imu_tau,
                                                                imu_w,
                                                                q) ) ) )**2
    
    part_2 = quaternion_norm(imu_a_qua - observation_model(q))**2
    return (part_1 + part_2)/2.

def sum_loss_func(q,imu_a_qua,imu_w,imu_tau):
    """
    aim:
    arg:
    return:
    """
    func_vectorized = jax.vmap(loss_func,(0,0,0,0,0))

    sum_result = jnp.sum(func_vectorized(q[:-1],q[1:],imu_a_qua[:-1],imu_w[:-1],imu_tau[:]))
    return sum_result

def get_array_norm(carry, q):
    """
    - `res`: The result from the previous loop.
    - `el`: The current array element.
    """
    q_norm = quaternion_norm(q)
    result = q/q_norm
    carry = q_norm
    return carry,result # ("carryover", "accumulated")
            # first one to the next carry
            # second one to the accumulator


def rots2euler(rots):
    result = np.zeros((rots.shape[2],3))
    for i in range(rots.shape[2]):
        result[i] = tf3d.euler.mat2euler(rots[:,:,i],axes = 'szyx')
    return result

def quat2euler(quaternions):
    result = np.zeros((quaternions.shape[0],3))
    for i,q in enumerate(quaternions):
        result[i] = tf3d.euler.quat2euler(q)
    return result

def quat2rots(quaternions):
    result = np.zeros((quaternions.shape[0],3,3))
    for i,q in enumerate(quaternions):
        result[i] = tf3d.quaternions.quat2mat(q)
    return result

# #testing the optimization with the second step:
# delta_q = grad_sum_loss_func(initial_pos, imud_a_qua, imud_w, tau_imud)
# q_next = initial_pos - alpha*delta_q
# fn, normlized_q = jax.lax.scan(get_array_norm, 0, q_next)


    # used for testing the NaN problem
# print(jax.jacrev(quaternion_norm)(jnp.asarray([1.0, 0.0, 0.0, 0.0])))
# print(jax.jacrev(quaternion_log)(jnp.asarray([1.0, 0.0, 0.0, 0.0])))


"""
    NOTE: the motion model initialization
    1) using jax.lax.scan  ( using rn )
    2) using for loop:taking too much time
    3) set quternions at all pos with [1,0,0,0]
"""

 














