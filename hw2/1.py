from itertools import product
import numpy as np
import pandas as pd


binary = (0, 1)

arrays = [
    [0, 0, 0, 0.25],
    [0, 0, 1, 0.05],
    [0, 1, 0, 0.05],
    [0, 1, 1, 0.20],
    [1, 0, 0, 0.05],
    [1, 0, 1, 0.15],
    [1, 1, 0, 0.10],
    [1, 1, 1, 0.15]]

joint = pd.DataFrame(arrays, columns=['x', 'y', 'z', 'p_xyz'])

# calculate marginals
p_x = joint.groupby(['x'])['p_xyz'].sum()
p_y = joint.groupby(['y'])['p_xyz'].sum()
p_z = joint.groupby(['z'])['p_xyz'].sum()

# calculate conditionals
p_x_given_y = joint.groupby(['y']).apply(
    lambda group: group.groupby(['x'])['p_xyz'].sum())
p_x_given_y = p_x_given_y / p_x_given_y.sum(axis=0)

p_y_given_x = joint.groupby(['x']).apply(
    lambda group: group.groupby(['y'])['p_xyz'].sum())
p_y_given_x = p_y_given_x / p_y_given_x.sum(axis=0)

p_y_given_z = joint.groupby(['z']).apply(
    lambda group: group.groupby(['y'])['p_xyz'].sum())
p_y_given_z = p_y_given_z / p_y_given_z.sum(axis=0)

p_z_given_y = joint.groupby(['y']).apply(
    lambda group: group.groupby(['z'])['p_xyz'].sum())
p_z_given_y = p_z_given_y / p_z_given_y.sum(axis=0)

p_y_given_xz = joint['p_xyz'].values.reshape(2, 2, 2).transpose(1, 0, 2)
p_y_given_xz = np.divide(
    p_y_given_xz,
    np.sum(p_y_given_xz, axis=0)[np.newaxis, :, :])

# 2.1 (a)(i)
total_kl = 0
for (x, y, z), outcome in joint.groupby(['x', 'y', 'z']):
    p_xyz = outcome['p_xyz'].values
    total_kl += p_xyz * np.log(p_xyz / (p_x[x] * p_y[y] * p_z[z]))
print('2.1(a)(i): ', total_kl)

# 2.1 (a)(ii)(a)
total_kl = 0
for x, y, z in product(binary, binary, binary):
    p_xyz = joint[(joint['x'] == x) & (joint['y'] == y) & (joint['z'] == z)]['p_xyz'].values
    factorized = p_x[x] * p_y_given_x.loc[y, x] * p_z_given_y.loc[z, y]
    total_kl += p_xyz * np.log(p_xyz / factorized)
print('2.1(a)(ii)(a): ', total_kl)


# 2.1 (a)(ii)(b)
total_kl = 0
for x, y, z in product(binary, binary, binary):
    p_xyz = joint[(joint['x'] == x) & (joint['y'] == y) & (joint['z'] == z)]['p_xyz'].values
    factorized = p_z[z] * p_y_given_z.loc[y, z] * p_x_given_y.loc[x, y]
    total_kl += p_xyz * np.log(p_xyz / factorized)
print('2.1(a)(ii)(b): ', total_kl)


# 2.1 (a)(ii)(c)
total_kl = 0
for x, y, z in product(binary, binary, binary):
    p_xyz = joint[(joint['x'] == x) & (joint['y'] == y) & (joint['z'] == z)]['p_xyz'].values
    factorized = p_y[y] * p_x_given_y.loc[x, y] * p_z_given_y.loc[z, y]
    total_kl += p_xyz * np.log(p_xyz / factorized)
print('2.1(a)(ii)(c): ', total_kl)


# 2.1 (a)(ii)(d)
total_kl = 0
for x, y, z in product(binary, binary, binary):
    p_xyz = joint[(joint['x'] == x) & (joint['y'] == y) & (joint['z'] == z)]['p_xyz'].values
    factorized = p_x[x] * p_z[z] * p_y_given_xz[y, x, z]
    total_kl += p_xyz * np.log(p_xyz / factorized)
print('2.1(a)(ii)(d): ', total_kl)
