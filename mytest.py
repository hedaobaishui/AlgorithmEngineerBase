""" 
# @Description: 
# @Author: 
# @Date: 2022-06-22 16:09:44
# @LastEditTime: 2022-06-23 16:12:08
# @LastEditors: taisanai
""" 
from matplotlib.font_manager import _Weight
import numpy as np
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
if __name__ == "__main__":
    # nms()
    test()
    # print(test())