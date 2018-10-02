from random import randint
import cv2
import sys
import os
import glob

def detect_faces(face_folder):
	for f in glob.glob(os.path.join(face_folder, "*.jpg")):
		
		inputFilepath = f
		filename_w_ext = os.path.basename(inputFilepath)
		filename, file_extension = os.path.splitext(filename_w_ext)
		
		image_path = f
		
		image=cv2.imread(image_path)
		
		image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

		faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
		count = 1
		for x,y,w,h in faces:
		
			file_path = output_folder + filename+'_'+str(count)
			
			sub_img=image[y-10:y+h+10,x-10:x+w+10]
			
			cv2.imwrite(file_path+'.jpg',sub_img)
			
			#cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)
			count += 1

		#cv2.imshow("Faces Found",image)
		#if (cv2.waitKey(0) & 0xFF == ord('q')) or (cv2.waitKey(0) & 0xFF == ord('Q')):
		#	cv2.destroyAllWindows()

if __name__ == "__main__":
	
	if not "Extracted_SOURCE" in os.listdir("."):
		os.mkdir("Extracted_SOURCE")
	if not "Extracted_TARGET" in os.listdir("."):
		os.mkdir("Extracted_TARGET")
    
	if len(sys.argv) != 3:
		print("Usage: python Detect_face.py 'source image folder' 'Extracted image folder'")
		sys.exit()
	images_SOURCE = sys.argv[1]
	Extracted_SOURCE = sys.argv[2]
	current_path = os.getcwd()	 
	CASCADE="Face_cascade.xml"
	FACE_CASCADE=cv2.CascadeClassifier(CASCADE)
	face_folder = current_path + '/' + images_SOURCE + '/'
	output_folder = current_path + '/' + Extracted_SOURCE + '/'
        
	detect_faces(face_folder)
