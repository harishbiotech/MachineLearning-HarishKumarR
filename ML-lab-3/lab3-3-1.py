#This is for my own understanding
# X=[[1,1],[1,3]]
# y=[[2],[6]]
# theta=[[0],[0]]
#
# def hypothesis(theta):
#     hypo=[]
#     for i in range(len(X)):
#         s=0
#         for j in range(len(X[0])):
#             for k in range(len(X)):
#                 s+=
#             o.append(theta[j][i]*X[j][i])
#         hypo.append(o)
#     return hypo
# hypo = hypothesis(theta)
# print(hypo)
# alpha=0.01
# def find_theta(X, y,alpha,hypo):
#       theta=[]
#       for i in range(len(X[0])):
#           s=0
#           for j in range(len(y)):
#               s+=((hypo[j][i]*X[j][i])-y[j][i])*X[j][i]
#           theta.append(hypo[i]-(alpha*s))
#       return theta
# t=find_theta(X,y,alpha,hypo)
# print(t)
# print("theta",t)
# def predict(X,theta):
#     # theta=find_theta(X,y,alpha)
#     prediction=[]
#     for i in range(len(X)):
#         s=0
#         for j in range(len(X)):
#             s+=theta[j]*X[i][j]
#         prediction.append(s)
#     return prediction
#prediction=predict(X)
#print("prediction",prediction)
# h=hypothesis(0)
# theta=find_theta(X,y,alpha,h)
# prediction=predict(X,theta)
# print("hypothesis",h)
# print("theta",theta)
# print("prediction",prediction)
# print("-------------------------------------------------------------")
#
# theta2 = find_theta(X, y, alpha, theta)
# prediction2 = predict(X, theta2)
# print("theta2", theta2)
# print("prediction2", prediction2)
# print("-------------------------------------------------------------")
# theta3 = find_theta(X, y, alpha, theta2)
# prediction3 = predict(X, theta3)
# print("theta3", theta3)
# print("prediction3", prediction3)
# print("--------------------------------------------------------------")
# theta4 = find_theta(X, y, alpha, theta3)
# prediction4 = predict(X, theta4)
# print("theta4", theta4)
# print("prediction4", prediction4)
# print("--------------------------------------------------------------")
# theta5 = find_theta(X, y, alpha, theta3)
# prediction5 = predict(X, theta5)
# print("theta5", theta5)
# print("prediction5", prediction5)
# print("--------------------------------------------------------------")
# theta6 = find_theta(X, y, alpha, theta2)
# prediction6 = predict(X, theta6)
# print("theta6", theta6)
# print("prediction6", prediction6)
# print("--------------------------------------------------------------")
# theta7 = find_theta(X, y, alpha, theta2)
# prediction7 = predict(X, theta7)
# print("theta7", theta7)
# print("prediction7", prediction7)
# print("--------------------------------------------------------------")
# theta8 = find_theta(X, y, alpha, theta3)
# prediction8 = predict(X, theta8)
# print("theta8", theta8)
# print("prediction8", prediction8)
# print("--------------------------------------------------------------")
# theta9 = find_theta(X, y, alpha, theta2)
# prediction9 = predict(X, theta9)
# print("theta9", theta9)
# print("prediction9", prediction9)
# print("--------------------------------------------------------------")
# theta10 = find_theta(X, y, alpha, theta3)
# prediction10 = predict(X, theta10)
# print("theta10", theta10)
# print("prediction10", prediction10)
# print("--------------------------------------------------------------")
X=[[1,1],[1,2]]
y=[[2],[4]]
theta=[[0],[0]]
def hypothesis(X,theta):
    hypo=[]
    for i in range(len(y)):
        s=0
        a=[]
        for j in range(len(theta[0])):
            for k in range(len(X)):
                s=s+theta[k][j]*X[j][k]
            a.append(s)
    hypo.append(a)
    for i in range(len(y)):
        s=0
        a=[]
        for j in range(len(theta[0])):
            for k in range(len(X)):
                s=s+theta[k][j]*X[1][k]
            a.append(s)
    hypo.append(a)
    return hypo
h=hypothesis(X,theta)
print("hypothesis",h)
alpha=0.001
def find_theta(X,y,h,alpha):
    thet=[]
    for i in range(len(y[0])):
       a=[]
       for j in range(len(h)):
            te=[]
            for k in range(len(h[0])):
                s=0
                #t=[]
                for l in range(len(h)):
                    s=s+(h[l][k]-y[l][k])*X[k][l]
                te.append(s)
       a.append(theta[i][i]-(alpha*te[i]))
       thet.append(a)
    for i in range(len(y[0])):
        a=[]
        for j in range(len(h)):
            te = []
            for k in range(len(h[0])):
                s = 0
                # t=[]
                for l in range(len(h)):
                    s = s + (h[l][k] - y[l][k]) * X[l][1]
                te.append(s)
        a.append(theta[1][i] - (alpha * te[i]))
        thet.append(a)
    return thet
te=find_theta(X,y,h,alpha)
print("thet",te)
for i in range(850.):
    print(f"{i}")
    theta=te
    hh=hypothesis(X,theta)
    print("hypo",hh)
    tt=find_theta(X,y,hh,alpha)
    print("theta",tt)
    print("-"*30)
    te=tt














































































