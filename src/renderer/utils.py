import pyrr
import numpy as np
def padded_vec3(v):
    return np.append(v, 0.0).astype(np.float32)
