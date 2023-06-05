import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

UPLOAD_FOLDER = r'D:\sinh vien fpt\sem7\DSS301\New folder\Face-Recognition-Attendance-System\IMAGE_FILES'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encoding_img(IMAGE_FILES):
    encodeList = []
    for img in IMAGE_FILES:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def takeAttendence(name):
    with open('attendence.csv', 'r+') as f:
        mypeople_list = f.readlines()
        nameList = []
        for line in mypeople_list:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            datestring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datestring}')

def gen():
    IMAGE_FILES = []
    filename = []
    dir_path = r'D:\sinh vien fpt\sem7\DSS301\New folder\Face-Recognition-Attendance-System\IMAGE_FILES'

    for imagess in os.listdir(dir_path):
        img_path = os.path.join(dir_path, imagess)
        img_path = face_recognition.load_image_file(img_path)  # reading image and append to list
        IMAGE_FILES.append(img_path)
        filename.append(imagess.split(".", 1)[0])

    encodeListknown = encoding_img(IMAGE_FILES)

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgc = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fasescurrent = face_recognition.face_locations(imgc)
        encode_fasescurrent = face_recognition.face_encodings(imgc, fasescurrent)

        for encodeFace, faceloc in zip(encode_fasescurrent, fasescurrent):
            matches_face = face_recognition.compare_faces(encodeListknown, encodeFace)
            face_distence = face_recognition.face_distance(encodeListknown, encodeFace)
            matchindex = np.argmin(face_distence)

            if matches_face[matchindex]:
                name = filename[matchindex].upper()
                y1, x2, y2, x1 = faceloc
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), 2, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                takeAttendence(name)

        cv2.imshow("campare", img)
        key = cv2.waitKey(20)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gen()
