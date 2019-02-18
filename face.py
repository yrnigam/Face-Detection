import cv2
#Face data xml file
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#continuous capture(0=inbuilt camera,1=ext camera)
video_capture = cv2.VideoCapture(0)

while True:
	ret, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#converting image to greyscale for faster loading	
	faces = faceCascade.detectMultiScale(
	gray,
	#reduce size to 10%(1.2=20%)
	scaleFactor=1.1,
	#no of faces
	minNeighbors = 5,
	minSize=(30,30)
	)
	
	#Green rectangle on face(RGB=0,255) thickness
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		
	cv2.imshow('Video',frame)
	#exiting the window
	if cv2.waitKey(1) & 0xFF  == ord('q'):  
		break
#free camera resource
video_capture.release()
#close video capture windows
cv2.destroyAllWindows()
