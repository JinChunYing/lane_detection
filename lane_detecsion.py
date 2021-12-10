#coding=gbk
#����ģ��
import cv2
import numpy as np

####��Ե��⺯��
def img_canny(frame):
    edge_img = cv2.Canny(frame,100,120)
    return edge_img
####�ɰ����
def roi_mask(edge_img):
    mask = np.zeros_like(edge_img) 
    mask = cv2.fillPoly(mask,np.array([[[0,380],[430,280],[600,280],[960,380]]]),color=255)
    masked_edge_img = cv2.bitwise_and(edge_img,mask)
    return masked_edge_img
####����任����ȡ������
def img2hough(masked_edge_img):
    lines = cv2.HoughLinesP(masked_edge_img,1,np.pi/180,15,minLineLength = 20,maxLineGap=100)
    return lines
####����ֱ��б��
def Calculate_slope(line):
    x_1,y_1,x_2,y_2 = line[0]
    return (y_2 - y_1)/(x_2 - x_1)
####�޳�б�ʲ���ϴ���߶�
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
####�������߶���ϳ�һ��
def least_squares_fit(lines):
    '''
    ��lines�е��߶���ϳ�һ���߶�
    param lines���߶μ��� [np.array([[x_1,y_1,x_2,y_2]]),np.array([[x_1,y_1,x_2,y_2]]),...,np.array([[x_1,y_1,x_2,y_2]])]
    return ���߶��ϵ����㣬np.array([[xmin,ymin],[xmax,ymax]])   '''
    #np.ravel ��ά��������һά
    #ȡ�����������
    x_coords = np.ravel([[line[0][0],line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1],line[0][3]] for line in lines])
    #np.polyfit ����ʽ��� k,b
    #����ֱ����ϣ��õ�����ʽϵ��
    poly = np.polyfit(x_coords,y_coords,deg=1)
    #np.polyval(poly,x)
    #���ݶ���ʽϵ������������ֱ���ϵĵ㣬����Ψһȷ������ֱ��
    point_min = (np.min(x_coords),np.polyval(poly,np.min(x_coords)))
    point_max = (np.max(x_coords),np.polyval(poly,np.max(x_coords)))
    return np.array([point_min,point_max],dtype=np.int)
#����������
def draw_line(img,left_points,right_points):
    if len(left_points) :
        cv2.line(img,tuple(left_points[0]),tuple(left_points[1]),color=(0,255,255),thickness=5)
    if len(right_points) :
        cv2.line(img,tuple(right_points[0]),tuple(right_points[1]),color=(0,255,255),thickness=5)
    return img
###################################################################
#������
#��ȡ��Ƶ
capture = cv2.VideoCapture("E:\\eclipse-workspace\\template-matching-ocr\\images\\lane_detection_01.mp4")
ret,frame = capture.read()
print(frame.shape) #��һ����Ƶ�Ŀ�ߣ�ȷ���ɰ�ѡ��Χ

while(True):
    ret , frame = capture.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edge_img = img_canny(frame_gray)        #����canny��Ե���
    masked_edge_img = roi_mask(edge_img)    #�ɰ������ѡȡ���ο�
    lines =img2hough(masked_edge_img)       #ѡȡ�����߶�
    #����б�ʴ���С���㣬��Ϊleft_lines��right_lines
    left_lines = [line for line in lines if Calculate_slope(line)>0]
    right_lines = [line for line in lines if Calculate_slope(line)<0]
    #�ֱ��޳������߶μ�����б�ʲ���ϴ���߶�
    left_lines = reject_abnormal_lines(left_lines,threshold=0.2)
    right_lines = reject_abnormal_lines(right_lines,threshold=0.2)
    #���ֱ��
    left_points = least_squares_fit(left_lines)
    right_points = least_squares_fit(right_lines)
    #��ֱ�߻���ͼƬ�У������߱�ע
    img = draw_line(frame,left_points,right_points)
    
    cv2.imshow('img',img)
    cv2.waitKey(5)

    