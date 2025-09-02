import cv2
from facerec_model import Facerecognition

#load camera
cap = cv2.VideoCapture(0)

#encode faces
sfr = Facerecognition()
sfr.load_images("images/")

while True:

    ret,frame = cap.read()

    #detect faces
    face_loci, face_name = sfr.detect_faces(frame)
    for face_loc, name in zip(face_loci, face_name):
        #print(face_loc)
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 -10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (8, 8, 288), 4)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()