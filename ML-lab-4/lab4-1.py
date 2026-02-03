import numpy as np

def gradient_descent(X_with_bias_train,y_np_train,h,alpha,theta_matrix):
    thet=[]
    for i in range(len(X_with_bias_train[0])):
        s=0
        for j in range(len(y_np_train)):
            s+=(h[j][0]-y_np_train[j][0])*X_with_bias_train[j][i]    # summation

        thet.append(theta_matrix[i][0]-(alpha*s))        # multiply by alpha and subtract with theta

    theta=np.array(thet)
    return theta.reshape(-1,1)