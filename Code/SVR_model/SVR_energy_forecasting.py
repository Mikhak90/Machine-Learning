import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Load your dataset
data = pd.read_csv('Code/SVR_model/input.csv')
data = data.values

# Initials
predicted_energy = np.array([])
data_rows = len(data[:,0])
NUM_HOURS = 24
TIME = range(1, NUM_HOURS + 1)

# do the prediction for every hour
for h in range(0,NUM_HOURS):
    #Convert X into n*1
    X_train = np.arange(1, data_rows + 1).reshape(-1, 1)
    Y_train = np.reshape(data[:,h],(data_rows,1))
    X_test = np.array([[data_rows + 1]]) #One day ahead
    
    # Create and train the SVR model
    svr = SVR(kernel='linear', C=1e3)
    svr.fit(X_train, Y_train)

    # Predict energy consumption for a new day
    predicted_energy = np.append(predicted_energy, svr.predict(X_test)) 

# Visualization (optional)
plt.plot(TIME, data[36,:], color='blue', label='Actual Data')
plt.plot(TIME, predicted_energy, color='red', label='SVR Prediction')
plt.xlabel('Time')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.show()

