import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from load_data import dataset
import utils
from main import initial_pos, opt_result_euler

from data_proccess import vicon_time,vicon_euler


vicon_time = np.reshape(vicon_time,(vicon_time.shape[1],))
initial_pos_euler = np.asarray(utils.quat2euler(initial_pos))
imud_time = np.reshape(utils.imud['ts'],(utils.imud['ts'].shape[1],))

# plot euler angle for x axis
plt.figure(figsize=(10, 5))
plt.plot(vicon_time, vicon_euler[:, 2], label='vicon data')
plt.plot(imud_time, opt_result_euler[:, 0], label='optimization result')
plt.plot(imud_time, initial_pos_euler[:, 0], label='zero step')
plt.xlabel('Time')
plt.ylabel('Angle (degrees)')
plt.title('mm_initialization x-axis Euler Angle over Time')
plt.legend()
plt.grid(True)
plt.savefig('mm_initialization x-axis_dataset '+dataset+ '.png')
plt.show()

# plot euler angle for y axis
plt.figure(figsize=(10, 5))
plt.plot(vicon_time, vicon_euler[:, 1], label='vicon data')
plt.plot(imud_time, opt_result_euler[:, 1], label='optimization result')
plt.plot(imud_time, initial_pos_euler[:, 1], label='zero step')
plt.xlabel('Time')
plt.ylabel('Angle (degrees)')
plt.title('mm_initialization y-axis Euler Angle over Time')
plt.legend()
plt.savefig('mm_initialization y-axis_dataset '+dataset+ '.png')
plt.grid(True)
plt.show()

# plot euler angle for z axis
plt.figure(figsize=(10, 5))
plt.plot(vicon_time, vicon_euler[:, 0], label='vicon data')
plt.plot(imud_time, opt_result_euler[:, 2], label='optimization result')
plt.plot(imud_time, initial_pos_euler[:, 2], label='zero step')
plt.xlabel('Time')
plt.ylabel('Angle (degrees)')
plt.title('mm_initialization z-axis Euler Angle over Time')
plt.legend()
plt.savefig('mm_initialization z-axis_dataset '+dataset+ '.png')
plt.grid(True)
plt.show()

