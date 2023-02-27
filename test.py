# -*- coding = utf-8 -*-
# @Time : 2023/2/10 10:55
# @Author : Peter
# @File : test.py
# @Software : PyCharm

import torch
alpha = torch.randn(3,1)
print(alpha)
best_score = (alpha >= 0).float()
print(best_score)
