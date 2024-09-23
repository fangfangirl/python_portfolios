import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def lppl(t, A, B, C, tc, beta, omega, phi):
    return np.exp(A + (B * (tc - t) ** beta) * (1+C * (np.cos(omega * np.log1p(tc - t) + phi)))).ravel()


def E(b1, A1, A, B, C, tc, beta, omega, phi):
    diff_1 = lppl(A1, A, B, C, tc, beta, omega, phi) - b1
    lppl_result = np.mean(diff_1**2)
    return lppl_result


def E_count(b1, A1, A, B, C, tc, beta, omega, phi):
    lppl_res = lppl(A1, A, B, C, tc, beta, omega, phi)
    print(lppl_res)
    diff_1 = lppl_res - b1
    lppl_result = np.mean(diff_1**2)
    return lppl_res, lppl_result


# 下載 APPL 歷史數據
symbol = 'AAPL'
start_date = '2023-01-01'
end_date = '2023-12-31'

data = yf.download(symbol, start=start_date, end=end_date)

# 選擇調整過後的收盤價
price_data = data['Adj Close']

# 將每一個收盤價取對數
log_price_data = np.log1p(price_data)

A1 = np.arange(0, len(log_price_data)).reshape((len(log_price_data), 1))
b1 = price_data

# 下一個程式部分
p = 10000  # 組群大小
r = 0.01  # 存活率
m = 10  # 突變數量
g = 10  # 演化代數


survive = round(p*r)  # 存活個體數目

pop = np.random.randint(0, 2, (p, 40))  # 初代，隨機的二進位基因族群,fu
fit = np.zeros((p, 1))

for generation in range(g):
    print(generation)
    # 將每個基因的適配度放入fit
    for pp in range(p):
        gene = pop[pp, :]  # 第p個基因
        # tc的數字範圍在[204,215]
        tc = (np.sum(2**np.array(range(10)) * gene[:10]) - 511) % 12 + 204
        # beta的數字範圍在[0,1]
        beta = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
        # omega的數字沒有範圍
        omega = (np.sum(2**np.array(range(10))*gene[20:30]))
        # phi的數字範圍在[0,2pi]
        phi = (np.sum(2**np.array(range(10)) * gene[30:40]) - 511) / 100
        # 將 phi 映射到區間 [0, 2pi]
        phi = max(0, min(2 * np.pi, phi))

        # F1的程式碼
        n = tc
        b1 = price_data[:n]
        A1 = np.arange(0, n).reshape((n, 1))
        b2 = np.zeros((n, 1))
        A2 = np.zeros((n, 3))  # 開A1二維陣列

        for i in range(n):
            b2[i] = log_price_data[i]
            A2[i, 0] = 1
            A2[i, 1] = (tc-i)**beta
            A2[i, 2] = ((tc-i)**beta) * np.cos(omega * np.log1p(tc - i) + phi)

        x = np.linalg.lstsq(A2, b2, rcond=None)[0]

        A = x[0]
        B = x[1]
        C = x[2] / B
        fit[pp] = E(b1, A1, A, B, C, tc, beta, omega, phi)
        # print(pp)

    # print("順序")
    sortf = np.argsort(fit[:, 0])  # 將裡面的數字由小到大的index獲取
    pop = pop[sortf, :]  # 使pop依照這個順序排列

    # 交配
    # print("交配")
    for i in range(survive, p):  # 從survive ~ p的基因死亡，產生孩子放回去
        # print("交配 : ", i)
        # 在前面還活者的選擇
        fid = np.random.randint(0, survive)  # 選擇父親
        mid = np.random.randint(0, survive)  # 選擇母親
        while mid == fid:  # 兩者要不同
            # print("hihi")
            mid = np.random.randint(0, survive)

        mask = np.random.randint(0, 2, (1, 40))
        son = pop[mid, :].copy()
        father = pop[fid, :]
        son[mask[0, :] == 1] = father[mask[0, :] == 1]  # 將子代mask為1的位置換成另一個父代的基因
        pop[i, :] = son

    # 突變
    # print("突變")
    for i in range(m):
        mr = np.random.randint(survive, p)
        mc = np.random.randint(0, 40)
        pop[mr, mc] = 1-pop[mr, mc]


# 計算十代之後留下的基因，計算各個基因的適配度
for i in range(p):
    gene = pop[i, :]
    tc = (np.sum(2**np.array(range(10)) * gene[:10]) - 511) % 12 + 204
    beta = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
    omega = (np.sum(2**np.array(range(10))*gene[20:30]))
    phi = (np.sum(2**np.array(range(10)) * gene[30:40]) - 511) / 100
    phi = max(0, min(2 * np.pi, phi))
    fit[i] = E(b1, A1, A, B, C, tc, beta, omega, phi)

sortf = np.argsort(fit[:, 0])
pop = pop[sortf, :]

gene = pop[0, :]  # 找到fit最低就是最好的
tc = (np.sum(2**np.array(range(10)) * gene[:10]) - 511) % 12 + 204
beta = (np.sum(2**np.array(range(10))*gene[10:20])-511)/100
omega = (np.sum(2**np.array(range(10))*gene[20:30]))
phi = (np.sum(2**np.array(range(10)) * gene[30:40]) - 511) / 100
phi = max(0, min(2 * np.pi, phi))

n = tc
b1 = price_data[:n]
A1 = np.arange(0, n).reshape((n, 1))
b2 = np.zeros((n, 1))
A2 = np.zeros((n, 3))  # 開A1二維陣列
for i in range(n):
    b2[i] = log_price_data[i]
    A2[i, 0] = 1
    A2[i, 1] = (tc-i)**beta
    A2[i, 2] = ((tc-i)**beta) * np.cos(omega * np.log1p(tc - i) + phi)

x = np.linalg.lstsq(A2, b2, rcond=None)[0]

A = x[0]
B = x[1]
C = x[2] / B

LPPL_final, fitness_MES = E_count(b1, A1, A, B, C, tc, beta, omega, phi)

print('tc:', tc, ' beta:', beta, ' omega:', omega, ' phi:', phi)
print('A:', A, ' B:', B, ' C:', C)
print("MES : ", fitness_MES)

tc_index = int(tc)
price_data_fit = price_data[:tc_index]

plt.plot(price_data_fit.index, price_data_fit, label='Actual Price Data')
plt.plot(price_data_fit.index, LPPL_final, label='Lppl Price Data')
plt.axvline(x=price_data.index[tc], color='red', linestyle='--', label='tc')

plt.title('LPPL Model Fit to Price Data')
plt.xlabel('Time')
plt.ylabel('Price_data')
plt.legend()
plt.show()
