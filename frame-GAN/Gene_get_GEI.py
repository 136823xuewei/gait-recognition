from skimage import io
import numpy as np
import cv2
import os

model_frames = 16
PATH = 'Generated_data_%d'%model_frames
NEW_PATH = 'Generated_Data_%d_GEI'%model_frames


def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.startswith("."):
                os.remove(os.path.join(root, name))
                print("Delete File: " + os.path.join(root, name))


def load_images_from_folder(folder):
    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), 0)
        if img is not None:
            images.append(img)

    return images


if not os.access(NEW_PATH, os.F_OK):
    os.mkdir(NEW_PATH)
del_files(PATH)

_count = 1

for _op in os.listdir(PATH):
    if not os.access(os.path.join(NEW_PATH, _op), os.F_OK):
        os.mkdir(os.path.join(NEW_PATH, _op))

    for human_id in os.listdir(os.path.join(PATH, _op)):
        if not os.access(os.path.join(NEW_PATH, _op, human_id), os.F_OK):
            os.mkdir(os.path.join(NEW_PATH, _op, human_id))
            
            for angle in os.listdir(os.path.join(PATH, _op, human_id)):
                os.mkdir(os.path.join(NEW_PATH, _op, human_id, angle))
                
                for _type in os.listdir(os.path.join(PATH, _op, human_id, angle)):
                    img_collection = load_images_from_folder(os.path.join(PATH, _op, human_id, angle, _type))  # load the images
                    frame_list = []
    
                    for image in img_collection:
                        # _count = _count + 1
                        # if _count % 2 == 0:
                        frame_list.append(image)
    
                    result = np.mean(frame_list, axis=0)
                    cv2.imwrite(os.path.join(NEW_PATH, _op, human_id, angle, '{}_GEI.png'.format(_type)), result)  # save the images
                
                print os.path.join(NEW_PATH, _op, human_id, angle)