import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
from typing import List
import math

data = np.loadtxt('LOB.csv', delimiter=',')
# Split data into columns
levels = (data[:, 0::2] // 100).astype(int)
volumes = (data[:,1::2]).astype(int)
ask_levels = levels[:, 0::2]
bid_levels = levels[:, 1::2]
ask_volumes = volumes[:,0::2]
bid_volumes = volumes[:,1::2]
# time = (message_data[:,0]*10**9).astype(np.int64) # Time of message

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)

def row_price(index):
    ask_level_row = ask_levels[index,:]
    bid_level_row = bid_levels[index,::-1]
    ask_volume_row = ask_volumes[index,:]
    bid_volume_row = bid_volumes[index,::-1]
    total_ask = 0
    ask_count = 0
    total_bid = 0
    bid_count = 0
    for i in range(len(ask_level_row)):
        total_ask = total_ask + (ask_level_row[i]*ask_volume_row[i])
        total_bid = total_bid + (bid_level_row[i]*bid_volume_row[i])
        ask_count += ask_volume_row[i]
        bid_count += bid_volume_row[i]
    return np.true_divide(total_ask, ask_count), np.true_divide(total_bid, bid_count)

def listDivide(myList):
    n = len(myList)//400
    new = [myList[i:i + n] for i in range(0, len(myList), n)]
    return new

normalized_ask = []
normalized_bid = []
for i in range(levels.shape[0]):
    current_ask, current_bid = row_price(i)
    normalized_ask.append(current_ask)
    normalized_bid.append(current_bid)

mean_ask_price = []
mean_bid_price = []
# Iterating over 1 minute windows in the day
new_normalized_ask = listDivide(normalized_ask)
new_normalized_bid = listDivide(normalized_bid)
for eachList in new_normalized_ask:
    mean_ask = np.mean(eachList)
    mean_ask_price.append(mean_ask)
for eachList in new_normalized_bid:
    mean_bid = np.mean(eachList)
    mean_bid_price.append(mean_bid)

print(mean_bid_price)
print(mean_ask_price)

def log_price(price_list, time_t):
    return math.log(np.abs(price_list[time_t]), 2)

def log_returns(price_list, current_time, unit_time):
    log_difference = log_price(price_list, current_time + unit_time) - log_price(price_list, current_time)
    return log_difference

ask_returns = []
for i in range(len(mean_ask_price) - 1):
    ask_returns.append(log_returns(mean_ask_price, i, 1))
# print(ask_returns)
bid_returns = []
for i in range(len(mean_bid_price) - 1):
    bid_returns.append(log_returns(mean_bid_price, i, 1))
# print(bid_returns)
plt.plot(list(range(len(ask_returns))), ask_returns, label='Ask')
plt.plot(list(range(len(bid_returns))), bid_returns, label='Bid')
plt.legend()
plt.show()

# set_seeds(8)
# n = OpinionNetwork(N=100, d=0.5, mu=0.5, type='albert-barabasi', ab_arg=[5])
# # nx.draw(n.g, with_labels=True)
# # plt.show()
# # Run evolution
# history = n.run_evolution(T=30)
# # Plot result
# n.plot_evolution()
list = [0, 1, 2, 3, 4, 5, 6]
print(list[1:6])
d = {'a': [1, 2], 'b': [4, 1], 'c': [2, 5], 'f': [0,12]}
# 第一种方法，key使用lambda匿名函数取value进行排序
a = sorted(d.items(), key=lambda x: x[1][1])
print(a)
b = '2238307.629744384'
print(round(float(b)))
d.pop('a')
print(a)
print(a[len(a)-1])
