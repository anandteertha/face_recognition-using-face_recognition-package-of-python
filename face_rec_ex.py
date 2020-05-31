import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.4
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn" #hog

print("loading known faces")

known_faces = []
known_names = []
for name in os.listdir(KNOWN_FACES_DIR):
	for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
		image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
		
		face_cascade = cv2.CascadeClassifier('data\haarcascade_frontalface_alt2.xml')
		
		image_ = cv2.imread(f"{KNOWN_FACES_DIR}/{name}/{filename}")

		gray = cv2.cvtColor(image_ , cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(gray, scaleFactor =1.5 , minNeighbors = 5)

		face = False
		for (x,y,w,h) in faces:
			
			if w > 0 :
				face = True
			
				print(face)
			else :
				print(False)
				pass

		print(name ,filename)

		if face ==True:
		
			encodings = face_recognition.face_encodings(image)[0]
			known_faces.append(encodings)
			known_names.append(name)










print("processing unknown faces")
count=0
for filename in os.listdir(UNKNOWN_FACES_DIR):
	print(filename)
	image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
	locations = face_recognition.face_locations(image,model=MODEL)
	encodings = face_recognition.face_encodings(image,locations)
	image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
	for face_encoding, face_location in zip(encodings, locations):
		results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
		match = None
		if True in results :
			match = known_names[results.index(True)]
			print(f"Match found: {match}")
			
			top_left = (face_location[3], face_location[0])
			bottom_right = (face_location[1] , face_location[2])
			color = [0,255,0]
			cv2.rectangle(image,top_left,bottom_right,color,FRAME_THICKNESS)
			top_left=(face_location[3],face_location[2])
			bottom_right=(face_location[1],face_location[2]+22)
			cv2.rectangle(image,top_left,bottom_right,color,cv2.FILLED)
			cv2.putText(image,match,(face_location[3]+10,face_location[2]+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),FONT_THICKNESS)
	cv2.imshow(filename,image)
	cv2.waitKey(0)
    	#cv2.destroyWindow(filename)