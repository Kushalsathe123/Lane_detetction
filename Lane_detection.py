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
    slopel=(left_line[3]-left_line[1])/(left_line[2]-left_line[0])
    right_line = coor(image,right_fit_avg)
    sloper=(right_line[3]-right_line[1])/(right_line[2]-right_line[0])
    print(slopel,sloper)
    if slopel<-0.9:
        print("hard left")
    elif sloper<0.5:
        print("left")
    else:
        print("straight")
    return np.array([left_line,right_line])
    
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    icanny = cv2.Canny(blur,50,400)
    return icanny

def draw_line(image,lines):
    image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(image,(x1,y1),(x2,y2),(255,0,0),10)
    return image

def roi(image): 
    width = image.shape[1]
    height = image.shape[0]
    polygons = np.array([[(0,height),(width,height),(width//2,height//2)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


cap = cv2.VideoCapture("video.mp4")
while(1):
    _,image = cap.read()
    image = cv2.resize(image,(960,540),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    icann = canny(image)
    cropped_image = roi(icann)
    lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=20,maxLineGap=10)
    avg_lines = avg_slope(image,lines)
    line_image = draw_line(image,avg_lines)
    combined_image = cv2.addWeighted(image,0.8,line_image,1,1)
    cv2.imshow("result",combined_image)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

