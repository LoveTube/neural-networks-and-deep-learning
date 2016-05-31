import numpy as np

#sigmoid function

def nonlin(x,deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))


x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

y = np.array([[0,0,0,1]]).T

np.random.seed(1)
sync0 = 2*np.random.random((3,1)) - 1

for iter in range(10000):
	l0 = x
	l1 = nonlin(np.dot(l0,sync0))
	
	l1_error = y -l1

	l1_delta = l1_error * nonlin(l1,True)

	sync0 += np.dot(l0.T,l1_delta)

print("Output Afer Training")
print(l1)



