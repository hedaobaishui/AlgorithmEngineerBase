<!--
 * @Description: 
 * @Author: 
 * @Date: 2022-04-27 22:15:42
 * @LastEditTime: 2022-06-17 11:17:00
 * @LastEditors: taisanai
-->
LoadWebcam
Loadstreams
[代码](/home/magic/AKApractice/YOLO3/yolov3-master/utils/datasets.py)
PIL.Image.open读入的是RGB顺序，而opencv中cv2.imread读入的是BGR通道顺序 
cv2.imread()
返回值是（height，width，channel）数组，channel的顺序是BGR顺序。
PIL.Image.open
需要用img=np.array(img)做转换，才能看到shape属性，是(height,width,channel)数组，channel的通道顺序为RGB。