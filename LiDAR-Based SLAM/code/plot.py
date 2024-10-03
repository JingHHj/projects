import numpy as np
import matplotlib.pyplot as plt
from sensors import states

x_coords = states[:,0]
y_coords = states[:,1]

plt.plot(x_coords, y_coords, color='blue', linestyle='-')

# labels and title
plt.title('Line Plot of Points')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

plt.show()