import pytesseract
import numpy as np
import cv2
import time
import pandas as pd
import os


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
states = {
    'AP': 'Andhra Pradesh',
    'AR': 'Arunachal Pradesh',
    'AS': 'Assam',
    'BR': 'Bihar',
    'CT': 'Chhattisgarh',
    'GA': 'Goa',
    'GJ': 'Gujarat',
    'HR': 'Haryana',
    'HP': 'Himachal Pradesh',
    'JH': 'Jharkhand',
    'KA': 'Karnataka',
    'KL': 'Kerala',
    'MP': 'Madhya Pradesh',
    'MH': 'Maharashtra',
    'MN': 'Manipur',
    'ML': 'Meghalaya',
    'MZ': 'Mizoram',
    'NL': 'Nagaland',
    'OD': 'Odisha',
    'PB': 'Punjab',
    'RJ': 'Rajasthan',
    'SK': 'Sikkim',
    'TN': 'Tamil Nadu',
    'TG': 'Telangana',
    'TR': 'Tripura',
    'UP': 'Uttar Pradesh',
    'UT': 'Uttarakhand',
    'WB': 'West Bengal'
}


def extract_num(img_name,folder_name):
    global read
    img = cv2.imread(folder_name+img_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in nplate:
        a,b = (int(0.02*img.shape[0]),int(0.025*img.shape[1]))
        plate = img[y+a:y+h-a, x+b:x+w-b,:]
        kernel = np.ones((1,1),np.uint8)
        plate = cv2.dilate(plate,kernel,iterations=1)
        plate = cv2.erode(plate,kernel,iterations=1)
        plate_gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        (thresh,plate) = cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY)

        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        print(read)
      #  try:
      #      print('Car Belongs to',states[stat])
       # except:
      #      print('State not recognised')
        # print(read)
        cv2.rectangle(img,(x,y),(x+w,y+h),(51,51,255),2)
        cv2.rectangle(img,(x,y-40),(x+w,y),(51,51,255),1)
        cv2.putText(img,read,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
        cv2.imshow('Plate',plate)

        df = pd.DataFrame({
                    'Time':[time.asctime(time.localtime(time.time()))],
                    'Vehicle Number':[read]
                },columns=['Time','Vehicle Number'])
        df.to_csv('DataSheet for Number.csv',mode='a',header = False)

    cv2.imshow('Result',img)
    #cv2.imwrite('result.jpg',img)
    cv2.waitKey(0)


if __name__ == '__main__':
    folder_name='Image\\'
    for root, dirs, files in os.walk(folder_name):
        for filesname in files:
            if filesname.endswith(('.jpg','.jpeg','.png')):
                extract_num(filesname,folder_name) 
