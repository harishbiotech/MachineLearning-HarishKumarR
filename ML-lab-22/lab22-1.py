# from ISLP import load_data
# from sklearn.cluster import PCA, HierarchicalClustering
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
#
# def hierarchial_clustering():
#     data = load_data('NCI60')
#     print(data)
#
# hierarchial_clustering()
import numpy as np
# C_0=0.5
# C_1=0.5
# C_0_S_0=0.5
# C_0_S_1=0.5
# C_1_S_0=0.9
# C_1_S_1=0.1
# C_0_R_0=0.8
# C_0_R_1=0.2
# C_1_R_0=0.2
# C_1_R_1=0.8
# W_1_S_0_R_0=0
# W_1_S_0_R_1=0.9
# W_1_S_1_R_0=0.9
# W_1_S_1_R_1=0.99
# W_0_S_0_R_0=1
# W_0_S_0_R_1=0.1
# W_0_S_1_R_0=0.1
# W_0_S_1_R_1=0.01
c=np.array([0.5,0.5])
s=np.array([0.5,0.5],
           [0.9,0.1])
r=np.array([0.8,0.2],
           [0.2,0.8])
