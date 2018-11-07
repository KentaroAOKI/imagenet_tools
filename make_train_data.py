import glob
import os
import random
import cv2

# parameters
from_dir = 'download_images'
to_dir = 'crop_images'
split_ratio = 0.75

def crop_image(src_image_path, dst_image_path):
    output_side_length=256
    img = cv2.imread(src_image_path)
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = int(output_side_length * height / width)
    else:
        new_width = int(output_side_length * width / height)
    resized_img = cv2.resize(img, (new_width, new_height))
    height_offset = int((new_height - output_side_length) / 2)
    width_offset = int((new_width - output_side_length) / 2)
    cropped_img = resized_img[height_offset : height_offset + output_side_length, width_offset : width_offset + output_side_length]
    cv2.imwrite(dst_image_path, cropped_img)

# make directory
if (os.path.exists(to_dir) == False):
    os.makedirs(to_dir)
    os.makedirs(os.path.join(to_dir, 'train'))
    os.makedirs(os.path.join(to_dir, 'test'))

# make list for train
train_list = open('train.txt','w')
test_list = open('test.txt','w')
label_list = open('labels.txt','w')

class_no=0
image_count = 0
labels = glob.glob('{}/*'.format(from_dir))
for label in labels:
    label_name = os.path.basename(label)
    print(label_name)
    os.makedirs(os.path.join(to_dir, 'train', label_name))
    os.makedirs(os.path.join(to_dir, 'test', label_name))
    images = glob.glob('{}/*.jpeg'.format(label))
    # write label for train
    label_list.write(label_name + '\n')
    length = len(images)
    split_count = 0
    split_number = length * split_ratio
    random.shuffle(images)
    for image in images:
        image_name = os.path.basename(image)
        if split_count < split_number:
            to_train_image = os.path.join(to_dir, 'train', label_name, image_name)
            print('{} > {}'.format(image, to_train_image))
            crop_image(image, to_train_image)
            # write image path for train
            train_list.write('{} {}\n'.format(to_train_image, class_no))
        else:
            to_test_image = os.path.join(to_dir, 'test', label_name, image_name)
            print('{} > {}'.format(image, to_test_image))
            crop_image(image, to_test_image)
            # write image path for test
            test_list.write('{} {}\n'.format(to_test_image, class_no))
        image_count = image_count + 1
        split_count = split_count + 1
    class_no += 1

train_list.close()
test_list.close()
label_list.close()