import numpy as np
import math

x = [13, 4, 11, 20]
y = [8, 5, 6, 15]

xm, ym = np.mean(x), np.mean(y)
print(np.mean(x))
print(np.mean(y))

chisl = 0
znam = 0

for k in range(len(x)):
    chisl+= (x[k] - xm)* (y[k] - ym)
    znam += (x[k] - xm)**2

print(chisl, znam, chisl/znam)

print(ym - xm* chisl/znam)