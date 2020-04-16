#forward propagation 
import numpy as np

# The Rectified Linear Activation Function
def relu(input):
    output = max(0,input)
    return output
    
input_data = np.array([2,3])

#store the weights of the nodes as dictionary 

weights = {
    'node_0': np.array([1,1]),
    'node_1': np.array([-1,1]),
    'output': np.array([2,-1])
}

node_0_input = (input_data * weights['node_0']).sum()
#activation func
node_0_output = relu(node_0_input)

node_1_input = (input_data * weights['node_1']).sum()
#activation 
node_1_output = relu(node_1_input)

hidden_layer_value = np.array([node_0_output, node_1_output])
output = (hidden_layer_value * weights['output']).sum()



print(hidden_layer_value)
print(output)