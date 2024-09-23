import numpy as np
import matplotlib.pyplot as plt


def F1(t):
    return 0.063*(t**3)-5.284*(t**2)+4.887*t+412+np.random.normal(0, 1)


def F2(t, A, B, C, D):
    return A*(t**B)+C*np.cos(D*t)+np.random.normal(0, 1, t.shape)


def E(b2, A2, A, B, C, D):
    return np.sum(abs(F2(A2, A, B, C, D)-b2))


# F1的程式碼
n = 1000
b1 = np.zeros((n, 1))
A1 = np.zeros((n, 5))  # 開A1二維陣列
for i in range(n):
    t = np.random.random()*100
    b1[i] = F1(t)
    A1[i, 0] = t**4
    A1[i, 1] = t**3
    A1[i, 2] = t**2
    A1[i, 3] = t
    A1[i, 4] = 1
x = np.linalg.lstsq(A1, b1, rcond=None)[0]
print(x)

# F2的程式碼
A2 = np.random.random((1000, 1))*100
b2 = F2(A2, 0.6, 1.2, 100, 0.4)

px1 = np.zeros((1024, 2))
for i in range(1024):
    D = (i-511)/100
    px1[i, 0] = D
    px1[i, 1] = E(b2, A2, 0.6, 1.2, 100, D)
plt.plot(px1[:, 0], px1[:, 1])
plt.show()

'''
px2 = np.zeros((1024, 1024, 3))
for i in range(1024):
    A = (i-511)/100
    for j in range(1024):
        C = j-511
        px2[i, j, 0] = A
        px2[i, j, 1] = C
        px2[i, j, 2] = E(b2, A2, A, 1.2, C, 0.4)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Using scatter to create a 3D scatter plot
ax.scatter(px2[:, 0], px2[:, 1], px2[:, 2])

# Setting labels
ax.set_xlabel('A')
ax.set_ylabel('C')
ax.set_zlabel('Absolute Error')

# Display the plot
plt.show()'''


# 下一個程式部分
p = 10000  # 組群大小
r = 0.01  # 存活率
m = 1000  # 突變數量
g = 10  # 演化代數

survive = round(p*r)  # 存活個體數目

pop = np.random.randint(0, 2, (p, 40))  # 初代，隨機的二進位基因族群,fu
fit = np.zeros((p, 1))

for generation in range(g):
    # 將每個基因的適配度放入fit
    for i in range(p):
        gene = pop[i, :]  # 第p個基因
        # A的數字範圍在[-5.11,5.12]
        A = (np.sum(2**np.array(range(10))*gene[:10])-511)/100
        # B的數字範圍在[-5.11,5.12]
        B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
        # C的數字範圍在[-511,512]
        C = np.sum(2**np.array(range(10))*gene[20:30])-511
        # D的數字範圍在[-5.11,5.12]
        D = (np.sum(2**np.array(range(10))*gene[30:40])-511)/100
        fit[i] = E(b2, A2, A, B, C, D)

    sortf = np.argsort(fit[:, 0])  # 將裡面的數字由小到大的index獲取
    pop = pop[sortf, :]  # 使pop依照這個順序排列

    # 交配
    for i in range(survive, p):  # 從survive ~ p的基因死亡，產生孩子放回去
        # 在前面還活者的選擇
        fid = np.random.randint(0, survive)  # 選擇父親
        mid = np.random.randint(0, survive)  # 選擇母親
        while mid == fid:  # 兩者要不同
            mid = np.random.randint(0, survive)

        mask = np.random.randint(0, 2, (1, 40))
        son = pop[mid, :].copy()
        father = pop[fid, :]
        son[mask[0, :] == 1] = father[mask[0, :] == 1]  # 將子代mask為1的位置換成另一個父代的基因
        pop[i, :] = son

    # 突變
    for i in range(m):
        mr = np.random.randint(survive, p)
        mc = np.random.randint(0, 40)
        pop[mr, mc] = 1-pop[mr, mc]

# 計算十代之後留下的基因，計算各個基因的適配度
for i in range(p):
    gene = pop[i, :]
    A = (np.sum(2**np.array(range(10))*gene[:10])-511)/100
    B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
    C = np.sum(2**np.array(range(10))*gene[20:30])-511
    D = (np.sum(2**np.array(range(10))*gene[30:40])-511)/100
    fit[i] = E(b2, A2, A, B, C, D)
sortf = np.argsort(fit[:, 0])
pop = pop[sortf, :]

gene = pop[0, :]  # 找到fit最低就是最好的
A = (np.sum(2**np.array(range(10))*gene[:10])-511)/100
B = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
C = np.sum(2**np.array(range(10))*gene[20:30])-511
D = (np.sum(2**np.array(range(10))*gene[30:40])-511)/100

print('A:', A, ' B:', B, ' C:', C, ' D:', D)
