import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle') This is just if you have a specific font style.

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")

i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show() #First graph

'''
Original:
w = 100
b = 100
These two variables were set to 100 at first but they did not fit in the correct line. So by changing the numbers in the two variables (w&b)
you can find the line that is best fit.
'''

w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

# To dynamically compute the line that best fits, you can create a function using the slope formula m = (y2-y1)/(x2-x1). In the calculate_slope() method
# you must pass on 2 arrays to calculate for the slope.

def calculate_slope(x, y):
    n = len(x)  # number of data points
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate the sums of squares
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    
    # Calculate the slope (m)
    slope = ss_xy / ss_xx
    
    return slope

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

#w = calculate_slope(x_train,y_train)     

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

w = 200 
#w = calculate_slope(x_train,y_train)      
b = 100
x_i = 1.2 #Compute output cost for a house that is 1200 sqft
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")