import numpy as np
import math

# Generate first 5 powers of 2
powersofTwo = []
for i in range(1,6):
  x = pow(2,i)
  powersofTwo.append(x)
  
# Generate powers of 2 between x and y
x = 3
y = 19
def powersofTwoBetwXY(x,y):
  p=np.int(np.ceil(math.log2(x))) # nearest power of 2 for x. in this eg: nearest power of 2 for 3 is 4. exponent of 4 for 2 power is 2
  q=np.int(np.ceil(math.log2(y))) 
  powersofTwo = []
  for i in range(p,q):
    x = pow(2,i) 
    powersofTwo.append(x)
  print(powersofTwo)
powersofTwoBetwXY(3,19)
# Output: [4, 8, 16]

  
  
  
  
