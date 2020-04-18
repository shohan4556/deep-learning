import numpy as np
from sklearn.metrics import mean_squared_error
import multiple_input as mi 

# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = mi.relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = mi.relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = mi.relu(input_to_final_layer)
    
    # Return model output
    return(model_output)

def main():
    # The data point you will make a prediction for
    input_data = np.array([[3,5], [1,-1], [0,0], [8,4]])

    # Sample weights
    weights_0 = {'node_0': [2, 1],
                'node_1': [1, 2],
                'output': [1, 1]
                }

    # Create weights that cause the network to make perfect prediction (3): weights_1
    weights_1 = {'node_0': [2, 1],
                'node_1': [1, 2],
                'output': [1, 0]
                }

    # The actual target value, used to calculate the error
    target_actual = [1,3,5,7]
    model_0_out = []
    model_1_out = []

    for row in input_data:
        model_0_out.append(predict_with_network(row, weights_0))
        model_1_out.append(predict_with_network(row, weights_1)) 
    
    mse_0 = mean_squared_error(target_actual, model_0_out)
    mse_1 = mean_squared_error(target_actual, model_1_out)
    print(mse_0)
    print(mse_1)

if __name__ == '__main__':
    main()
    