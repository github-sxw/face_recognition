import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(
    r'D:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    r'D:\ProgramData\Anaconda3\Lib\site-packages\cv2\data\haarcascade_eye_tree_eyeglasses.xml')
IMAGE_SIZE = 64


def get_image_from_video(save_num, file_name):
    cap = cv2.VideoCapture(0)
    os.chdir('C:/Users/Administrator/Pictures/trans_face_data/'+file_name)
    num_counter = 1
    while num_counter <= save_num:
        status, img = cap.read()
        if not status:
            print('摄像头异常，强制推出！！')
            break
        img_w, img_h, _ = img.shape
        img_vet = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_loc = face_cascade.detectMultiScale(img_vet, scaleFactor=1.2, minSize=(100, 100))
        if len(faces_loc) != 0:
            x, y, w, h = faces_loc[0]
            if w > 150:
                x_ = 0 if (x - 10 < 0) else x - 10
                y_ = 0 if (y - 10 < 0) else y - 10
                image_name = '{}.jpg'.format(num_counter)
                image_file = img[y_:y+h+10, x_:x+w+10]
                cv2.imwrite(image_name, image_file)
                num_counter += 1
            cv2.rectangle(img, (x, y), (x+w, y+h), color=(128, 125, 0), thickness=2)
            # cv2.imshow('video', img[y-10:y+h+10, x-10:x+w+10])
        cv2.imshow('video', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def get_image_from_folder(root_path):
    num_counter = 1
    os.chdir('C:/Users/Administrator/Pictures/')
    for root, dirs, files in os.walk(root_path):
        if files is []:
            continue
        else:
            for image_name in files:
                image_path = os.path.join(root, image_name)
                img = cv2.imread(image_path)
                img_w, img_h, _ = img.shape
                # print(img.shape)
                img_vet = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces_loc = face_cascade.detectMultiScale(img_vet, scaleFactor=1.2, minSize=(100, 100))
                if len(faces_loc) != 0:
                    x, y, w, h = faces_loc[0]
                    if w > 150:
                        image_file_name = '{}.jpg'.format(num_counter)
                        x_ = 0 if (x-10 < 0) else x-10
                        y_ = 0 if (y-10 < 0) else y-10
                        image_file = img[y_:y + h + 10, x_:x + w + 10, :]

                        # cv2.imshow('video', image_file)
                        # print(img)
                        cv2.imwrite('other/'+image_file_name, image_file)
                        print(image_file_name)
                        num_counter += 1
                    cv2.rectangle(img, (x, y), (x + w, y + h), color=(128, 125, 0), thickness=2)
                cv2.imshow('video', img)
                k = cv2.waitKey(1)
                if k == 27:
                    return cv2.destroyAllWindows()


def load_image_data(folder_path):
    os.chdir('C:/Users/Administrator/Pictures/')
    image_array_list = []
    lable_list = []
    for lable in os.listdir(folder_path):
        lable_full_path = os.path.abspath(os.path.join(folder_path, lable))
        for image in os.listdir(lable_full_path):
            image_array = cv2.imread(os.path.join(lable_full_path, image))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            resize_image_array = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))
            resize_image_array = resize_image_array.astype('float32')
            # print(resize_image_array.shape)
            resize_image_array /= 255
            # cv2.imshow(image, resize_image_array)
            # cv2.waitKey(0)
            image_array_list.append(resize_image_array)
            if lable == 'sxw':
                lable_list.append(1)
            elif lable == 'baobao':
                lable_list.append(2)
            else:
                lable_list.append(0)
    return image_array_list, lable_list


if __name__ == '__main__':
    # get_image_from_video(2000, 'baobao')
    # get_image_from_folder('faces.zip')
    image_array_list, lable_list = load_image_data("trans_face_data")
