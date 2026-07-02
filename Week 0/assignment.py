import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import time 

def load_data(data_path):
    f =open(data_path)
    arr = f.read().replace(',',' ').split()
    arr = np.array(arr,  dtype='float')
    arr = arr.reshape(-1,2)
    return arr
def initialise_centers(data, K, init_centers=None):
    if(init_centers==None):
        return data[random.choice(np.arange(len(data)), size=K, replace=False)]
    return init_centers
def initialise_labels(data):
    temp = np.arange(len(data))
    return temp
def calculate_distances(data, centers):
    temp = data.reshape(-1,1,2)
    temp1 = centers.reshape(1,-1,2)
    temp = temp - temp1
    temp = np.sqrt(np.sum(temp**2,axis=2))
    return temp
def update_labels(distances):
    x1 = np.min(distances, axis=1).reshape(len(distances),1)
    x1 = x1[:,np.newaxis]==distances[:,np.newaxis]
    x1 = np.where(x1[:,np.newaxis]==True)
    return x1[3]
def update_centers(data, labels, K):
    centers = np.array([np.mean(data[labels==t],axis=0) for t in range(K)])
    return centers
def check_termination(labels1, labels2):
    return np.all(labels1==labels2)
def kmeans(data_path:str, K:int, init_centers):
    '''
    Input :
        data (type str): path to the file containing the data
        K (type int): number of clusters
        init_centers (type numpy.ndarray): initial centers. shape = (K, 2) or None
    Output :
        centers (type numpy.ndarray): final centers. shape = (K, 2)
        labels (type numpy.ndarray): label of each data point. shape = (N,)
        time (type float): time taken by the algorithm to converge in seconds
    N is the number of data points each of shape (2,)
    '''
    data = load_data(data_path)    
    centers = initialise_centers(data, K, init_centers)
    labels = initialise_labels(data)

    start_time = time.time() # Time stamp 

    while True:
        distances = calculate_distances(data, centers)
        labels_new = update_labels(distances)
        centers = update_centers(data, labels_new, K)
        if check_termination(labels, labels_new): break
        else: labels = np.copy(labels_new)
 
    end_time = time.time() # Time stamp after the algorithm ends
    return centers, labels, end_time - start_time 
data_path = 'spice_locations.txt'
K, init_centers = 2, None
data = load_data(data_path) 
centers, labels, time_taken = kmeans(data_path, K, init_centers)
print('Time taken for the algorithm to converge:', time_taken)
def visualise(data_path, labels, centers):
    data = load_data(data_path)
    plt.figure(figsize=(9, 6))
    plt.scatter(data[:,0],data[:,1], color='blue', label='Data Points')
    plt.scatter(centers[:,0],centers[:,1], color='red', s=50, marker='o', label='K-centers')
    plt.title("K-means clustering")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.savefig("kmeans.png", dpi=300, bbox_inches='tight')
    plt.show()
    return plt
visualise(data_path, labels, centers)