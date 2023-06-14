#Author: Gustavo Rubio
#Date: 06/01/2023
#MSEIP-AFRL Internship Summer 2023
#This code is a modification of the HousePriceModel.py file you can use this files for testing.
import numpy as np
import matplotlib.pyplot as plt

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)

'''
Original values:
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
'''

#These two new train variables are for testing
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show() #First graph

# Calculate the slope (w) using the calculate_slope() function
def calculate_slope(x, y):
    n = len(x)  # number of data points
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate the sums of squares
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    
    # Calculate the slope (w)
    slope = ss_xy / ss_xx
    
    return slope

w = calculate_slope(x_train, y_train)

# Set the intercept (b) manually or compute it using mean values
b = np.mean(y_train) - w * np.mean(x_train)

print(f"w: {w}")
print(f"b: {b}")

# Compute the model output using the updated w and b
def compute_model_output(x, w, b):
    return w * x + b

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show() #Second graph

x_i = 1.2 #Compute output cost for a house that is 1200 sqft
cost_1200sqft = compute_model_output(x_i, w, b)

print(f"${cost_1200sqft:.0f} thousand dollars")

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost

total_cost = compute_cost(x_train, y_train, w, b)

print(f"The cost of this model is: {total_cost}")
