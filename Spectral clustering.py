import numpy as np  
import matplotlib.pyplot as plt



dataset=np.loadtxt('SpectData.txt')



# Wieghts_at_segma = 0.1 , 0.01 , 0.05

dataset_copy=dataset

w=[]
for i in range(2084):
    dd=[]
    for j in dataset_copy:
        m=np.linalg.norm(np.array(dataset[i])-np.array(j))
        d=np.exp(-m**2/(2*(0.1**2)))
        
        dd.append(d)
    w.append(dd)
        

w=np.array(w) 
np.fill_diagonal(w,0)

#Degrees

x=[]
for i in w:
    x.append(sum(i))
       
t=np.diag(x)


#Graph_Laplacian
L=t-w


#Sorting eigenvectors according to its eigenvalues

eVls, eVcs=np.linalg.eig(L)
x = np.argsort(eVls)
eVls = eVls[x]
eVcs = eVcs[:,x]


#Clustering the data


y=eVcs[:,1]


plt.scatter(dataset[y < 0, 0], dataset[y < 0, 1],c="red")
plt.scatter(dataset[y > 0, 0], dataset[y > 0, 1],c="black")

plt.show()

    
