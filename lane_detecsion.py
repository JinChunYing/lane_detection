#coding=gbk
#导入模块
import cv2
import numpy as np

####边缘检测函数
def img_canny(frame):
    edge_img = cv2.Canny(frame,100,120)
    return edge_img
####蒙版操作
def roi_mask(edge_img):
    mask = np.zeros_like(edge_img) 
    mask = cv2.fillPoly(mask,np.array([[[0,380],[430,280],[600,280],[960,380]]]),color=255)
    masked_edge_img = cv2.bitwise_and(edge_img,mask)
    return masked_edge_img
####霍夫变换，提取所有线
def img2hough(masked_edge_img):
    lines = cv2.HoughLinesP(masked_edge_img,1,np.pi/180,15,minLineLength = 20,maxLineGap=100)
    return lines
####计算直线斜率
def Calculate_slope(line):
    x_1,y_1,x_2,y_2 = line[0]
    return (y_2 - y_1)/(x_2 - x_1)
####剔除斜率差异较大的线段
def reject_abnormal_lines(lines,threshold):
    slopes = [Calculate_slope(line) for line in lines]
    while len(lines)>0:
        mean = np.mean(slopes)
        diff = [abs(s-mean) for s in slopes]
        idx = np.argmax(diff)
        if diff[idx]>threshold:
            lines.pop(idx)
            slopes.pop(idx)
        else :
            break
    return lines
####将多条线段拟合成一条
def least_squares_fit(lines):
    '''
    将lines中的线段拟合成一条线段
    param lines：线段集合 [np.array([[x_1,y_1,x_2,y_2]]),np.array([[x_1,y_1,x_2,y_2]]),...,np.array([[x_1,y_1,x_2,y_2]])]
    return ：线段上的两点，np.array([[xmin,ymin],[xmax,ymax]])   '''
    #np.ravel 高维数组拉成一维
    #取出所有坐标点
    x_coords = np.ravel([[line[0][0],line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1],line[0][3]] for line in lines])
    #np.polyfit 多项式拟合 k,b
    #进行直线拟合，得到多项式系数
    poly = np.polyfit(x_coords,y_coords,deg=1)
    #np.polyval(poly,x)
    #根据多项式系数，计算两个直线上的点，用于唯一确定这条直线
    point_min = (np.min(x_coords),np.polyval(poly,np.min(x_coords)))
    point_max = (np.max(x_coords),np.polyval(poly,np.max(x_coords)))
    return np.array([point_min,point_max],dtype=np.int)
#画出车道线
def draw_line(img,left_points,right_points):
    if len(left_points) :
        cv2.line(img,tuple(left_points[0]),tuple(left_points[1]),color=(0,255,255),thickness=5)
    if len(right_points) :
        cv2.line(img,tuple(right_points[0]),tuple(right_points[1]),color=(0,255,255),thickness=5)
    return img
###################################################################
#主函数
#读取视频
capture = cv2.VideoCapture("E:\\eclipse-workspace\\template-matching-ocr\\images\\lane_detection_01.mp4")
ret,frame = capture.read()
print(frame.shape) #看一下视频的宽高，确定蒙版选择范围

while(True):
    ret , frame = capture.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edge_img = img_canny(frame_gray)        #调用canny边缘检测
    masked_edge_img = roi_mask(edge_img)    #蒙版操作，选取矩形框
    lines =img2hough(masked_edge_img)       #选取所有线段
    #根据斜率大于小于零，分为left_lines和right_lines
    left_lines = [line for line in lines if Calculate_slope(line)>0]
    right_lines = [line for line in lines if Calculate_slope(line)<0]
    #分别剔除左右线段集合中斜率差异较大的线段
    left_lines = reject_abnormal_lines(left_lines,threshold=0.2)
    right_lines = reject_abnormal_lines(right_lines,threshold=0.2)
    #拟合直线
    left_points = least_squares_fit(left_lines)
    right_points = least_squares_fit(right_lines)
    #将直线画到图片中，车道线标注
    img = draw_line(frame,left_points,right_points)
    
    cv2.imshow('img',img)
    cv2.waitKey(5)

    