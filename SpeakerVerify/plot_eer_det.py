# -*- coding: utf-8 -*-                                                                                                                 

import numpy as np
import matplotlib.pyplot as plt
import sys
# compute FRR FAR
score_and_target = open(sys.argv[1], 'r').readlines()
target_score = []
nontarget_score = []

FRR = []
FAR = []


for line in score_and_target:
    score, isTarget = line.strip().split()
    if isTarget == "target":
        target_score.append(float(score))
    else:
        nontarget_score.append(float(score))

target_score.sort()
nontarget_score.sort()

print target_score
print nontarget_score

P_cnt = len(target_score)
N_cnt = len(nontarget_score)


x_y=[]
y_x=[]

for index in range(40):
    fr = float(index)*100/P_cnt
    print fr
    FRR.append(fr)
    x_y.append(fr)
    y_x.append(fr)
    FP=0
    for index_2 in range(N_cnt):
        if target_score[index] < nontarget_score[index_2]:
            FP += 1
    FAR.append(float(FP)*100/N_cnt)



print FRR
print FAR



            
        
plt.plot(FRR, FAR)
plt.plot(x_y, y_x)
plt.show()
