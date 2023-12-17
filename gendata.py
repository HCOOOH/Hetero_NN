import csv
import random
from collections import defaultdict

import numpy as np
from scipy.stats import beta, norm

goods = ["A", "B", "C", "D", "E"]
flow_for_goods = {
    "A":0.3,
    "B":0.1,
    "C":0.2,
    "D":0.05,
    "E":0.1
}

quantity_for_goods = {
    "A":10,
    "B":7,
    "C":5,
    "D":3,
    "E":5
}

profit_for_goods = {
    "A":0.2,
    "B":0.4,
    "C":0.1,
    "D":0.3,
    "E":0.5
}

class Company:

    def __init__(self, head, arrange):
        self.head = head
        scale = 5
        if self.head == "A":
            p_mean = 3000
            b_mean = 10000
            j_mean = 2000
            rate = np.random.uniform(0.6, 0.8)
        elif self.head == "B":
            p_mean = 1500
            b_mean = 4000
            j_mean = 1000
            rate = np.random.uniform(0.4, 0.6)
        else:
            p_mean = 500
            b_mean = 1000
            j_mean = 300
            rate = np.random.uniform(0.3, 0.5)

        self.time = np.random.normal(8, 4)

        self.z1 = np.random.normal(p_mean, 5000, arrange).mean()
        self.z2 = np.random.normal(b_mean, 5000, arrange).mean()
        self.z3 = np.random.normal(j_mean, 500, arrange).mean()
        # self.z2 = mean * np.random.beta(a=2, b=8, size=arrange).mean()
        # self.z3 = mean * np.random.beta(a=5, b=5, size=arrange).mean()

        self.flow = (self.z1 + self.z2 + self.z3) / self.time / 60
        self.client_num = self.z1 * (1 - norm.cdf(3, 20, 5)) + self.z2 * (
                1 - beta.cdf(0.8, 2, 8) + self.z3 * (1 - beta.cdf(0.8, 5, 5)))
        self.watch_time = (self.z1 * norm.mean(20, 5) + self.z2 * beta.mean(2, 8) * scale + self.z3 * beta.mean(5,
                                                                                                                5)) * scale / (
                                      self.z1 + self.z2 + self.z3)
        self.point_num = self.z1 * rate
        self.share_num = self.z2 * rate
        print(norm.cdf(3, 20, 5))

    def print(self):
        print(
            f"z1 = {self.z1}, z2 = {self.z2}, z3 = {self.z3}, flow = {self.flow}, client_num = {self.client_num}, watch_time = {self.watch_time}")


class Merchant:
    def __init__(self,T, F):
        self.total_profit = 0.0
        self.goods_num = np.random.randint(1,len(goods))
        random.shuffle(goods)
        self.goods = goods[0:self.goods_num]
        # self.goods_dict = defaultdict(list)
        self.time = T
        self.flow = F
        self.quanlity = {}
        for good in self.goods:
            self.quanlity[good] = np.random.uniform(0.8,0.95)


    def cul_profit(self, T):
        for good in self.goods:
            self.total_profit += (profit_for_goods[good] * T[good]) * (self.flow * flow_for_goods[good] + quantity_for_goods[good]) * self.quanlity[good] * self.time
        print(self.total_profit)

    def print(self):
        print(
            f"goods = {self.goods}, flow = {self.flow}, time = {self.time}, quanlity = {self.quanlity}, ")


company = Company("A", 1)
company.print()
merchant = Merchant(company.time,company.flow)
merchant.print()
T = {
    "A":1,
    "B":1,
    "C":1,
    "D":1,
    "E":1
}
merchant.cul_profit(T)


