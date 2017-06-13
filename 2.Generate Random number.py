import random as rn
import numpy as np
import time

#------------------------------------------#

#1.產生五個亂數，並將其輸出

x = [rn.randint(1,100) for i in range(5)]
print('Random 5 numbers:',x)

#Random 5 numbers: [37, 73, 91, 86, 53]

#------------------------------------------#

#2.產生N個介於-1與1之間的亂數，計算其平均值與標準差並輸出，每個亂數的值則不用輸出。
#  N=10^1, 10^2, 10^3, 10^4, 10^5。

def Generator(N):
    X = [(rn.random()-0.5)*2 for i in range(N)]
    Mean = sum(X)/N
    SD = np.std(X)
    print(N, 'numbers', 'Mean:', Mean, 'SD:',SD)


N = [10**1,10**2, 10**3, 10**4, 10**5]

for num in N:
    Generator(num)

#10 numbers Mean: 0.05178784441788187 SD: 0.44390855751
#100 numbers Mean: 0.04754108308199876 SD: 0.59291694929
#1000 numbers Mean: 0.015035609675168931 SD: 0.573246749249
#10000 numbers Mean: 0.0012145369755242636 SD: 0.57800505946
#100000 numbers Mean: -0.0018908851449471427 SD: 0.577392754609

#------------------------------------------#

#1.做基本題2時，一併輸出產生每N個亂數前後的系統時間，並計算所需的時間
def Generator_plusT(N):
    T = time.time()
    X = [(rn.random()-0.5)*2 for i in range(N)]
    Mean = sum(X)/N
    SD = np.std(X)
    T_ = time.time()
    print(N, 'numbers', 'Mean:', Mean, 'SD:',SD,'Time Cost:', T_-T)

for num in N:
    Generator_plusT(num)

#10 numbers Mean: -0.08389043449574181 SD: 0.533774820799 Time Cost: 0.00031304359436035156
#100 numbers Mean: 0.02795960848186926 SD: 0.576011349053 Time Cost: 0.00020003318786621094
#1000 numbers Mean: -0.01638346568225237 SD: 0.571972062387 Time Cost: 0.0004112720489501953
#10000 numbers Mean: 0.007783556468448909 SD: 0.576357992259 Time Cost: 0.0035347938537597656
#100000 numbers Mean: 0.0017431266199803052 SD: 0.577092602473 Time Cost: 0.0422358512878418

#------------------------------------------#

#2.自己寫一個亂數產生器

def Random():  #Return a random number between 0 and1.
    B = time.time()                 #Use current computer time as random variable.
    #Convert the integer B to string and split the integer and decimal numbers for later usage.
    B_str = str(B).split('.') 
    #get the string length in order to get a return between 0 & 1      
    L = len(B_str[1])
    #fetch the decimal numbers (which is more likely to change when we call it.)
    B = int(B_str[1])

    # If the number is even , reverse it in order to get more random result.
    if B % 2 ==0:
        B = B_str[1][::-1]
        B = int(B)/10**L       # Set the number between 0 and 1
    else:
        B /= 10**L             # Set the number between 0 and 1

    return B                   # Return Result

def myGenerator(N):
    T = time.time()
    X = [Random() for i in range(N)]
    Mean = sum(X)/N
    SD = np.std(X)
    T_ = time.time()
    print(N, 'numbers', 'Mean:', Mean, 'SD:',SD,'Time Cost:', T_-T)

N = [10**1,10**2, 10**3, 10**4, 10**5,10**6]

for num in N:
    myGenerator(num)

#10 numbers Mean: 0.6920977200000001 SD: 0.226436565572 Time Cost: 0.0001761913299560547
#100 numbers Mean: 0.714055922 SD: 0.169871872636 Time Cost: 0.0005848407745361328
#1000 numbers Mean: 0.6829826293 SD: 0.212686428153 Time Cost: 0.004462242126464844
#10000 numbers Mean: 0.7037589384500031 SD: 0.226297003833 Time Cost: 0.06615591049194336
#100000 numbers Mean: 0.4493022169219989 SD: 0.298093849493 Time Cost: 0.5950288772583008
#1000000 numbers Mean: 0.5199096325592232 SD: 0.278055646016 Time Cost: 5.914984941482544



