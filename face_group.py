import os
import dlib
import glob
import cv2


current_path = os.getcwd()
#model_path = current_path + '/models/'
shape_predictor_model = 'shape_predictor_68_face_landmarks.dat'
face_rec_model =  'dlib_face_recognition_resnet_model_v1.dat'
face_folder = current_path + '/images_SOURCE/'
output_folder = current_path + '/Face_Group_output_folder/'


detector = dlib.get_frontal_face_detector()
shape_detector = dlib.shape_predictor(shape_predictor_model)
face_recognizer = dlib.face_recognition_model_v1(face_rec_model)


descriptors = []
images = []

for f in glob.glob(os.path.join(face_folder, "*.jpg")):

    print (f)
    img = cv2.imread(f)

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    dets = detector(img2, 1)
    print("Number of faces detected: {}".format(len(dets)))


    for index, face in enumerate(dets):

        shape = shape_detector(img2, face)

        face_descriptor = face_recognizer.compute_face_descriptor(img2, shape)


        descriptors.append(face_descriptor)
        images.append((img2, shape))
print (descriptors)


labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
print("labels: {}".format(labels))
num_classes = len(set(labels))
print("Number of clusters: {}".format(num_classes))

face_dict = {}
for i in range(num_classes):
    face_dict[i] = []
# print face_dict
for i in range(len(labels)):
    face_dict[labels[i]].append(images[i])

for key in face_dict.keys():
    file_dir = os.path.join(output_folder, str(key))
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    for index, (image, shape) in enumerate(face_dict[key]):
        file_path = os.path.join(file_dir, 'face_' + str(index))
        print file_path
        dlib.save_face_chip(image, shape, file_path, size=150, padding=0.25)
