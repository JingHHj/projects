import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from load_data import imud,camd,dataset #,vicd
import transforms3d as tf3d
import utils
from main import opt_result_rots
# from data_proccess import vicon_euler,vicon_rotst

# vicon: (3, 3, 5561)

# rots = vicon_rots
# ts = np.asarray(vicd['ts']) 
    # extract the vicon time information
camd_time = np.asarray(camd['ts'].flatten())
    # extract the cam time information

ts = np.asarray(imud['ts'])
rots = opt_result_rots

print(camd_time.shape)
print(ts.shape)
print(rots.shape)

# testdata1: 
# camd_time: (1685,)
# ts: (1, 5645)
# rots(5645, 3, 3)
# result (3,3,1685)

def get_useful_rots(sample_ts,object_ts,object_data):
    """
    aim: for everytime step in camd dataset we will find a correspond time step in the object dataset (either imu or vicon)
        then we just extract the data in those time
    arg:
    return:
    """
    result_data = np.zeros(( 3, 3, sample_ts.shape[0]))
    for i in range(sample_ts.shape[0]):
        idx = np.abs(sample_ts[i] - object_ts).argmin()
        # print(idx)
        # print(i)
        result_data[:,:,i] = object_data[idx,:,:]
    return result_data

  
useful_rots = get_useful_rots(camd_time,ts,rots)
cam_data = camd['cam']  # (240, 320, 3, 1685)
cam_data = cam_data/255.


x, y = np.meshgrid(np.arange(320),
                   np.arange(240))
og_coords = np.vstack([x.flatten(),y.flatten()]).transpose()  # (76800, 2)

sph_coords = np.vstack([np.full(og_coords.shape[0],1),
                        og_coords[:,0]*(np.pi/3)/320-np.pi/6,
                        og_coords[:,1]*(np.pi/4)/240-np.pi/8]).transpose()
        # (r, az, elev)

cart_coords = np.zeros((sph_coords.shape[0], 3)) # (76800, 3, 3)


def sph2cart(spherical_coords):
    r, h, v  = spherical_coords[0], spherical_coords[1], spherical_coords[2]
    x = r * np.cos(v) * np.cos(h)
    y = r * np.cos(v) * np.sin(h)
    z = r * np.sin(v)
    return np.array([x, y, z])

def cart2sph(cartesian_coords):
    x, y ,z = cartesian_coords[0], cartesian_coords[1], cartesian_coords[2]
    r = (x**2+ y**2+ z**2)**0.5
    v = jnp.arcsin(z/r)
    h = jnp.arctan2(y,x)
    return jnp.array([r, h, v])


for i,j in enumerate(sph_coords):
    cart_coords[i] = sph2cart(j)

# cart_coords (76800, 3)
# useful_rots (3, 3, 1685)

total_frame = np.zeros((useful_rots.shape[2],3,cart_coords.shape[0]))
for i in range(useful_rots.shape[2]):
    total_frame[i,:,:] = np.dot(useful_rots[:,:,i],cart_coords.T)

# total_frame (1685, 3, 76800)

# vectorized_cart2sph = np.vectorize(cart2sph)

# sph_total_frame = np.zeros((useful_rots.shape[2],2,cart_coords.shape[0]))
# for i in range(total_frame.shape[0]):
#     temp = total_frame[i,:,:].T
#     sph_total_frame[i,:,:] = vectorized_cart2sph(temp).T
 

total_frame = jnp.array(total_frame)


vectorized_cart2sph = jax.vmap(cart2sph, in_axes=(0,), out_axes=0)

# using the vectorized function on the data
sph_total_frame = vectorized_cart2sph(total_frame)

sph_total_frame = sph_total_frame[:,1:,:] # without the radius
    # (1685, 2, 76800)


sph_total_frame = np.array(sph_total_frame)


sph_total_frame[:,0,:] = (sph_total_frame[:,0,:]+ np.pi)*1920/(2*np.pi)
sph_total_frame[:,1,:] = (sph_total_frame[:,1,:]+ np.pi/2)*960/np.pi


final_frame = np.round(sph_total_frame).astype(int)
    # (1685, 2, 76800)

# print(final_frame[:100,:,:1])

# camd:(y,x,intensity,t)

# canvas = np.zeros((1920,960,3))
canvas = np.zeros((1000,2000,3))
# og_coords (76800, 2)

for i in range( final_frame.shape[0] ):
    for j in range( final_frame.shape[2] ):
        canvas[final_frame[i,1,j],final_frame[i,0,j],:] = cam_data[og_coords[j,1], og_coords[j,0], : , i]
    
plt.imshow(canvas)
plt.savefig('opt_panorama_'+dataset+ '.png')
plt.show()



# for i in range(cam_data.shape[3]):
#     plt.imshow(cam_data[:,:,:,i])
#     # plt.show()



    # canvas[ final_frame[i,:,:] ] = cam_data[og_coords, : , i]
# h = [0], horizontal
# v = [1], vertical
# 360/60*320 = 1920
# 180/45*240 = 960



# h*1920/(2*pi)+pi
# v*1000/pi+pi/2




    



