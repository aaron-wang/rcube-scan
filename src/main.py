import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

def myContour(colorName, mask):
    contours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if (len(contours) >= 1):
        c_threshold = 500
            # c_threshold = 1000
        for c in contours:
            if (cv.contourArea(c) > c_threshold):
                x, y, w, h = cv.boundingRect(c)
                if (colorName == "red"):
                    if (w > 50 or h > 50): continue
                    if (w*h > 2500):
                        continue
                    # else:
                        # print(f"{w} | {h}")
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                # cv.drawContours(frame,c,-1,(0,0,255),3)
                cv.putText(frame,colorName,(x,y+h),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

# bob = 96

while True:
    ret,frame = cap.read()
    hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV) 

    # Yellow color
    
    # OK 
    # lyellow = np.array([22,60,40])
    # uyellow = np.array([40,255,255])
    # BETTER
    # lyellow = np.array([20,96,40])
    # uyellow = np.array([80,120,255])
    # lyellow = np.array([15,80,40])
    # uyellow = np.array([35,255,255])
    lyellow = np.array([20,50,40])
    uyellow = np.array([70,120,255])
    y_mask = cv.inRange(hsv_frame,lyellow,uyellow)
    yellow = cv.bitwise_and(frame,frame,mask=y_mask)
    # contours, hierarchy = cv.findContours(y_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # if (len(contours) >= 1):
    #     for c in contours:
    #         if (cv.contourArea(c) > 500):
    #             x, y, w, h = cv.boundingRect(c)
    #             cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #             # cv.drawContours(frame,c,-1,(0,0,255),3)
    #             cv.putText(frame,"yellow",(x,y+h),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    # # green color
    lgreen = np.array([50,20,67])
    hgreen = np.array([110,255,100])
    g_mask = cv.inRange(hsv_frame,lgreen,hgreen)
    green = cv.bitwise_and(frame,frame,mask=g_mask)


    #blue
    lblue = np.array([110,40,10])
    hblue = np.array([120,150,255])
    b_mask = cv.inRange(hsv_frame,lblue,hblue)
    blue = cv.bitwise_and(frame,frame,mask=b_mask)
    

    #red
    # lred = np.array([10,40,90])
    # hred = np.array([40,100,140])
    # lred = np.array([37,30,70])
    # hred = np.array([55,70,100])
    # lred = np.array([0,100,30])
    # hred = np.array([25,180,90])
    
    lred = np.array([0,70,50])
    hred = np.array([8,255,180])
    lred2 = np.array([170,70,50])
    hred2 = np.array([180,255,100])

    # lred3 = np.array([170,70,50])
    # hred3 = np.array([180,255,255])


    r_mask = cv.inRange(hsv_frame,lred,hred)
    r_mask2 = cv.inRange(hsv_frame,lred2,hred2)
    r_mask = r_mask | r_mask2
    red = cv.bitwise_and(frame,frame,mask=r_mask)

    myContour("yellow",y_mask)
    # cv.imshow('yellow mask',yellow)

    myContour("green",g_mask)
    # cv.imshow('green mask',green)

    myContour("blue",b_mask)
    # cv.imshow('blue mask',blue)

    myContour("red",r_mask)
    cv.imshow("red mask", red)





    cv.imshow('frame',frame)

    # HS V - hue, saturation, value (brightness)
    key = cv.waitKey(1)
    if (key == ord('q')): 
        break
    if (key == ord('w')):
        bob+=1
        print(bob)
    if (key == ord('d')): 
        bob-=1
        print(bob)

cap.release()
cv.destroyAllWindows()