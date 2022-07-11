'''
Author: hedaobaishui 896585355@qq.com
Date: 2022-06-22 16:09:44
LastEditors: hedaobaishui 896585355@qq.com
LastEditTime: 2022-07-11 10:59:01
FilePath: /cavaface-master/home/magic/AKApractice/akaNotes/mytest.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
""" 
# @Description: 
# @Author: 
# @Date: 2022-06-22 16:09:44
# @LastEditTime: 2022-06-23 16:12:08
# @LastEditors: taisanai
""" 
# from matplotlib.font_manager import _Weight
import numpy as np
import torch.nn as nn
import torch
def nms(dets,thresh):
    #计算交并比　排序　迭代
    keep = []
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    areas = (y2-y1+1)*(x2-x1+1)
    order = np.argsort(dets[:,4])[::,-1] #从大到小排列
    while order.size>0:
        centerbox_i = order[0]
        keep.append(centerbox_i)
        xx1 =  np.max(x1[centerbox_i],x1[order[1:]])
        xx2 =  np.min(x2[centerbox_i],x2[order[1:]])
        yy1 =  np.max(y1[centerbox_i],y1[order[1:]])
        yy2 =  np.max(y2[centerbox_i],y2[order[1:]])
        
        w = np.max(xx2 - xx1, 0)
        h = np.max(yy2 - yy1, 0)
        ovr = w*h / ((x2[centerbox_i]-x1(centerbox_i))*(y2[centerbox_i]-y1[centerbox_i])+\
        (x2[order[1:]]-x1[order[1:]])*(y2[order[1:]]-y1[order[1:]])-w*h)
        ovr = w*h / ((areas[centerbox_i])+\
            areas[order[1:]]-w*h)
        #　删除小于阈值的
        index = np.where(ovr<thresh)[0]
        order=order[index+1]
    return keep
def softnms(dets,thresh,softmethod):
    #已经获得得到ovr
    ovr = np.array([1,2,3,4,5])
    if softmethod == 1:
        if ovr > thresh:
            weight = 1 - ovr
        else:
            weight = 1
    elif softmethod == 2:
        if ovr > thresh:
            weight = np.exp((-ovr*ovr)/0.5)
    elif softmethod == 3:
        if ovr > thresh:
            weight = 0
        else:
            weight = 1
    dets[:,4] = dets[:,4] * weight
    #通过不断的迭代　降低　score来筛选
    return 
def test():
    ll = [1,2,3,4,5]
    index = np.where(np.array(ll)>3)[0][::]
    print(np.array(ll)[index])
def testtorchloss():
    ll = []
    for i in range(10):
        tmp = []
        for j in range(10):
            if i==j:
                tmp.append(1)
            else:
                tmp.append(0)
        ll.append(tmp)
    true = torch.tensor(ll,dtype=torch.float32)
    pre = torch.randn(10,10)
    prelogit = torch.softmax(pre,dim=1)

    print("asdsaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    print(torch.sum(prelogit,dim=1))

    # print(1-prelogit)
    # manuloss = (-true*torch.log(prelogit)-(1-true)*(torch.log(1-prelogit))).mean()
    manuloss = (-true*torch.log(prelogit)).sum()/10

    crossloss_ = nn.CrossEntropyLoss(reduce='mean')
    crossloss = crossloss_(pre,true)
    # bcloss_ = nn.BCEWithLogitsLoss(reduce='mean')
    # bcloss = bcloss_(pre,true)
    print(true,pre)
    print(manuloss)
    print(crossloss)
    # print(bcloss)
    for i in range(0,1000,20):
        logp = torch.tensor(i/100.)
        p = torch.exp(-logp)
        # 交叉熵越大　表示越难预测　导致最终的loss越大－
        loss = (1-p) ** 2 * logp
        print(logp,loss)

def sum0(f,s,nums,out):
    mid = (f+s)//2
    while f<s:
        mid = (f+s)//2
        while mid < s and mid > f:
            if nums[f]+nums[s]+nums[mid]<0:
                mid = (mid + s) // 2
                print("<",f,mid,s)
                if mid == s - 1:
                    f += 1
                    break
            elif nums[f]+nums[s]+nums[mid]>0:
                mid = (f + mid) // 2 
                print(">",f,mid,s)
                if mid == f-1:
                    s -= 1
                    break
            else:
                out.add((nums[f],nums[mid],nums[s]))
                print(out)
                sum0(f+1,s,nums,out)
                sum0(f,s-1,nums,out)
    return out
def threeSum( nums )  :
    nums.sort()
    out = set()
    sum0(0,len(nums)-1,nums,out)
    return out

def testsequential(x):
    seq = nn.Sequential()
    x1 = seq(x)
    print(x1-x)
if __name__ == "__main__":
    # # nms()
    # nums = [-1,0,1,2,-1,-4]
    # b = 17754*3 - 4000*6 + 24500 * 3
    # c = [17754*21, 24500*14 +27500*9]
    # a = [17520,113515.52,35000,8000,2500,4103.21,3028.28,17754,17754*2,10000]
    # print(type(a))
    # print(type(a).__name__)
    # # threeSum(nums)
    # # print(test())
    xx = torch.randn((3,3,224,224))
    print(xx)
    testsequential(xx)
