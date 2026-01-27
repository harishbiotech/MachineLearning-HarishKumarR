#This is for my own understanding
X=[[1,1,2],[1,2,1]]
y=[3,4]

def hypothesis(hyp):
    hypo=[]
    for i in range(len(X)):
        for j in range(len(X[0])):
            hypo.append(hyp*X[i][j])
        return hypo
hypothes=hypothesis(0)
print(hypothes)
#alpha=0.01
# def find_theta(X, y,alpha):
#     theta=[]
#     for i in range(len(y)):
#         s=0
#         for j in range(len(X[0])):
#             summestion=(hypothes[i][j]-y[i][j])
#             s=s+summestion
#             theta.append(hypothes[i][j-1]-(alpha*summestion))
#         return theta
# print(find_theta(X, y, alpha))




