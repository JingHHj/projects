import jax.numpy as jnp
import jax
import numpy as np
import transforms3d as tf3d 
from load_data import imud
import utils

tau_imud = jnp.array(np.diff((imud['ts']))).transpose() # ()
    # get the tau,which is the differences between the current time step and next time step
raw_imud = jnp.array(imud['vals']).transpose()
    # get the a and w from the origial data


imu_biases = utils.get_biases(raw_imud)  
    # extracting the biases data from the first 50 data
calibrated_imud = utils.get_calibrated(raw_imud,imu_biases)
    # calibrated the raw imu data
calibrated_imud = utils.flip_ax_ay(calibrated_imud)
    # flip the a on x and y ( add negivitve sign)


from load_data import vicd
vicon_rots = vicd['rots'] #(3, 3, 5561)
    # extract vicon data ( from a dictionary )
vicon_time = np.asarray(vicd['ts']) 
    # extract the vicon time information
vicon_euler = utils.rots2euler(vicon_rots)
    # convert vicon data (rotation matrices in zyx) into euler angles (a 5561*3 matrices)  


