import numpy as np 
import jax
import load_data
import utils
import jax.numpy as jnp 
import transforms3d as tf3d 
import utils
import data_proccess


q_0 = jnp.array([1.,0.,0.,0.])  # the zero pos
    # motion model initialize
mm_initial_imud = jnp.hstack((data_proccess.calibrated_imud[:-1, 3:],data_proccess.tau_imud))
    # put calibrated_imud and tau_imud together for jax.lax.scan use
l, inital_pos = jax.lax.scan(utils.mm_initial, q_0, mm_initial_imud)
    # getting the motion model initialization reasult
initial_pos = jnp.vstack([q_0,inital_pos]) # add the zero pos to intial pos
    # initial_pos shape: (5644,4)

# initial_pos = jnp.tile(q_0,(data_proccess.calibrated_imud.shape[0],1)) 
#     # initializing all the zero pos by setting them [1,0,0,0]


imud_a, imud_w = data_proccess.calibrated_imud[:,:3], data_proccess.calibrated_imud[:,3:]
    # extract a and w from imud, for easier use in loss function
imud_a_qua = jnp.insert(imud_a, 0, 0, axis=1) 
    # turned imud_a (1*3 vector) into a array of quaternion
grad_sum_loss_func = jax.jacobian(utils.sum_loss_func)
    # doing gradient to the summation of the loss function

    # initial_pos: (5645, 4)   # imud_a_qua: (5645, 4)  # imud_w: (5645, 3)  # tau_imud: (5644, 1)  # delta_q shape: (5645,4)

alpha = 0.0001
    # defining step size
ts = jnp.arange(2700)
    # define time step

def optimization(initial_q, ts):
    """
    - `res`: The result from the previous loop.
    - `el`: The current array element.
    """
    delta_q = grad_sum_loss_func(initial_q, imud_a_qua, imud_w, data_proccess.tau_imud)
    q_next = initial_q - alpha* delta_q
    x,normlized_q_next = jax.lax.scan(utils.get_array_norm, 0, q_next)
    
    return normlized_q_next, normlized_q_next  # ("carryover", "accumulated")

opt_result, _ = utils.jax.lax.scan(optimization, initial_pos, ts)


# opt_result = np.asarray(opt_result)
    # convert optimization result (quaternions in xyz) into euler angles
opt_result_euler = utils.quat2euler(np.asarray(opt_result))
    # putting the optimization result into np.array so we can deal with it with for loop

opt_result_rots = utils.quat2rots(np.asarray(opt_result))