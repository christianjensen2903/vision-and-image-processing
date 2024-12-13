# -*- coding: utf-8 -*-
"""
Run Beethoven reconstruction code 

Author: Francois Lauze, University of Copenhagen
Date: Mon Jan  4 14:11:54 2016
"""


import numpy as np
import ps_utils
import numpy.linalg as la
import matplotlib.pyplot as plt
from typing import Literal

# read Beethoven data
I, mask, S = ps_utils.read_data_file("buddha")
method: Literal["ransac", "inverse", "pseudo"] = "ransac"


# get indices of non zero pixels in mask
nz = np.where(mask > 0)
m, n = mask.shape

# for each mask pixel, collect image data
n_points = len(S)
J = np.zeros((n_points, len(nz[0])))
for i in range(n_points):
    Ii = I[:, :, i]
    J[i, :] = Ii[nz]


if method == "ransac":
    M_list = []
    for i in range(J.shape[1]):
        J_i = J[:, i : i + 1]
        mi, _, _ = ps_utils.ransac_3dvector((J_i, S), 10, verbose=0)
        M_list.append(mi)

    M = np.hstack(M_list)
elif method == "inverse":
    iS = la.inv(S)
    M = np.dot(iS, J)
else:
    iS = la.pinv(S)
    M = np.dot(iS, J)


# solve for M = rho*N
iS = la.pinv(S)
M = np.dot(iS, J)

# get albedo as norm of M and normalize M
Rho = la.norm(M, axis=0)
N = M / np.tile(Rho, (3, 1))

n1 = np.zeros((m, n))
n2 = np.zeros((m, n))
# n3 = np.full((m, n), 0)
n3 = np.zeros((m, n))
n1[nz] = N[0, :]
n2[nz] = N[1, :]
n3[nz] = N[2, :]

_, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(n1)
ax2.imshow(n2)
ax3.imshow(n3)
plt.show()

z = ps_utils.unbiased_integrate(n1, n2, n3, mask)
# z = np.nan_to_num(z)
z = np.nan_to_num(z, nan=np.nanmin(z))

ps_utils.display_surface(z)
