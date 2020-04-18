# updated_weight = weights - learning_rate * slop 
# slop = slop of loss func * input_data * error 
# error = target - predicted
# update the model multiple times 

'''
- start at random weight 
- forward propagation to make prediction 
- backward propagation to calculate slope of the loss function 
- multiply the slope by learning rate and subtract from the current weight 
'''

import numpy as np 
import matplotlib.pyplot as plt 

#return slop of loss function 
def get_slope(input_data, target, weights):
    error = get_error(input_data, target, weights)
    slope = 2 * input_data * error 
    return slope

#return mean square error 
def get_mse(input_data, target, weights):
    error = get_error(input_data, target, weights)
    mse = np.mean(error **2)
    return mse

#return slope 
def get_error(input_data, target, weights):
    preds = (input_data * weights).sum()
    error = preds - target
    return error

# The data point you will make a prediction for
input_data = np.array([4,2,3])

#target 
target_actual = 0

# Sample weights
weights = np.array([10,1,2])

learning_rate = 0.01

mse_hist = []

#print(get_slope(input_data, target_actual, weights))

for i in range(20):
    slope = get_slope(input_data, target_actual, weights)
    weights = weights - learning_rate * slope
    print('iteration {0} weights : {1}'.format(i, weights))
    mse = get_mse(input_data, target_actual, weights)
    mse_hist.append(mse)

plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()



