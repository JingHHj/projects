import jax.numpy as jnp
from jax import lax
import numpy as np


a = jnp.array([
    [0.1, 10],
    [0.2, 5],
    [0.4, 2.5],
    [0.5, 2],
    [0.8, 1.25],
    [1,  1]
    ])

def cumsum(res, el):
    """
    - `res`: The result from the previous loop.
    - `el`: The current array element.
    """
    result = res + el[0]*el[1]
    return result, result  # ("carryover", "accumulated")
result_init = 0
final, result = lax.scan(cumsum, result_init, a)

# print(result,"\n",final)

# print(float(1e-6))




x, y = np.meshgrid(np.arange(240),
                   np.arange(320))
og_coords = np.vstack([x.flatten(),y.flatten()]).transpose()


# canvas = np.zeros((960,1920,3))

# sph_total_frame = jnp.random.random((1685, 2, 76800))
# print(sph_total_frame) # (1685, 2, 76800)

# x = sph_total_frame

# len = sph_total_frame.shape

# cam_data = camd['cam']  # (240, 320, 3, 1685)

# for i in range(len[0]):
#     for j in range(len[2]):
#        canvas[x[i,:,j]] = cam_data[og_coords[j,0],og_coords[j,1],:,i]

