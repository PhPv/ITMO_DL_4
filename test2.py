import sympy as sym

# s1, s2, s3, s4
# a41, a31, a11, a12, a22, a33

v1, v2, v3, v4 = sym.Symbol("v1"), sym.Symbol("v2"), sym.Symbol("v3"), sym.Symbol("v4")
y = 0.8
p11, p12, P112, P113, R112, R113, P213, R213 = 0.6, 0.4, 0.3, 0.7, 2, 3, 1, 3
p22, P221, R221 = 1, 1, 3
p31, p33, P131, R131, P333, P334, R333, R334 = 0.3, 0.7, 1, -3, 0.2, 0.8, 1, 6
p41, P141, P142, R141, R142 = 1, 0.6, 0.4, 5, -3

a221 = P221 * (R221 + y * v1)
a213 = P213 * (R213 + y * v3)
a333 = P333 * (R333 + y * v3)
a334 = P334 * (R334 + y * v4)
a112 = P112 * (R112 + y * v2)
a131 = P131 * (R131 + y * v1)
a142 = P142 * (R142 + y * v2)
a141 = P141 * (R141 + y * v1)
a113 = P113 * (R113 + y * v3)

# v1 = p11 * (a112 + a113) + p12 * a213
# v2 = a221
# v3 = p31 * a131 + p33 * (a333 + a334)
# v4 = a142 + a141

print(p11 * (a112 + a113) + p12 * a213)
print(p22 * a221)
print(p31 * a131 + p33 * (a333 + a334))
print(p41 * (a142 + a141))

solution = sym.solve(
    (
        p11 * (a112 + a113) + p12 * a213,
        p22 * a221,
        p31 * a131 + p33 * (a333 + a334),
        p41 * (a142 + a141)
    ),
    v1,
    v2,
    v3,
    v4,
)
print(solution)


import pandas as pd
import math

d = {'Type': ['Email', 'Words'], 'spam': [19, 145], 'ham': [21, 112]}
info  = pd.DataFrame(data=d)
d2 = {'Word': ['Unlimited', 'Coupon', 'Free', 'Offer', 'Prize', 'Access', 'Bill', 'Cash', 'Online', 'Gift'], 
      'spam': [5,3,0,0,5,8,23,47,2,52], 
      'ham': [8,2,10,6,18,1,8,7,11,41]}
words = pd.DataFrame(data=d2)