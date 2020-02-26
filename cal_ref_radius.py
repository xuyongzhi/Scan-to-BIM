import math

def get_g(dis, radius):
  ref = pow(radius,2) * 2
  return  math.exp(- dis / ref )

radius = 4
for d in range(10):
   g = get_g(pow(d,2), radius)
   print(f'd:{d}  g:{g}')
