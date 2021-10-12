#ce code a pour but d'importer mon modéle, et ensuite ajouter openCV avec pour visualiser
#i will have a big work to do here
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
#cascade de haar pour la detection de visage
face_classifier = cv2.CascadeClassifier(r'C:\Users\itta_\Documents\python_projects\FaceEmotion\haarcascade_frontalface_default.xml')
#load my model
classifier =load_model(r'C:\Users\itta_\Documents\python_projects\FaceEmotion\model.h5')
#label my output
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


#what will comes next is a standard approche that came from the tutoriel of OpenCV, i just added the
#the label of the ouput emotion
cap = cv2.VideoCapture(0)



while True:
    res, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #because my model was trained on Grey Scale images
    faces = face_classifier.detectMultiScale(gray)
    #OPENCV tuto
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        region_gray = gray[y:y+h,x:x+w]
        region_gray = cv2.resize(region_gray,(48,48),interpolation=cv2.INTER_AREA) #psk mon modéle est trained sur 48*48



        if np.sum([region_gray])!=0:
            rey = region_gray.astype('float')/255.0
            rey = img_to_array(rey)
            rey = np.expand_dims(rey,axis=0)

            prediction = classifier.predict(rey)[0]
            
            label=emotion_labels[prediction.argmax()] #pour que le label me donne le max
            label_position = (x,y-15)# -15 psk si je mets 10 ça va rester coller
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        else:
            cv2.putText(frame,'No Face',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Detection Emotion',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()