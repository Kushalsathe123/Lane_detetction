import cv2
import numpy as np

def coor(image,line_parameters):
    slope,intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1- intercept)/slope)
    x2 = int((y2- intercept)/slope)

    return np.array([x1,y1,x2,y2])

def avg_slope(image,lines):
    left_fit=[]
    right_fit =[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_avg = np.average(left_fit,axis=0)
    right_fit_avg = np.average(right_fit,axis=0)
    left_line = coor(image,left_fit_avg)
    right_line = coor(image,right_fit_avg)
    return np.array([left_line,right_line])
    


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    icanny = cv2.Canny(blur,50,150)
    return icanny

def draw_line(image,lines):
    image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(image,(x1,y1),(x2,y2),(255,0,0),10)
    return image

def roi(image): 
    height = image.shape[0]
    polygons = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image
# def allroi(image):
    
#     height = image.shape[0]
#     width = image.shape[1]
#     polygons = np.array([[(0,height),(width,height),(550,250)]])
#     mask = np.zeros_like(image)
#     cv2.fillPoly(mask,polygons,255)
#     masked_image = cv2.bitwise_and(image,mask)
#     return masked_image

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
icann = canny(lane_image)
cropped_image = roi(icann)
lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
avg_lines = avg_slope(lane_image,lines)
line_image = draw_line(lane_image,avg_lines)
combined_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)
cv2.imshow("result",combined_image)
cv2.waitKey(0)
