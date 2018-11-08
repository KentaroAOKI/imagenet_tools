import argparse
import chainer
from chainer.cuda import to_cpu
import chainer.functions as F
import cv2
import numpy as np
from PIL import Image
import random

# import alex
# import googlenet
# import googlenetbn
# import nin
import resnet50
# import resnext50

def load_crop_image(src_image_path):
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
    return cropped_img

def predict_resnet50(net, x):
    h = net.bn1(net.conv1(x))
    h = F.max_pooling_2d(F.relu(h), 3, stride=2)
    h = net.res2(h)
    h = net.res3(h)
    h = net.res4(h)
    h = net.res5(h)
    h = F.average_pooling_2d(h, 7, stride=1)
    h = net.fc(h)    
    return F.softmax(h)

def main():
    archs = {
        # 'alex': alex.Alex,
        # 'alex_fp16': alex.AlexFp16,
        # 'googlenet': googlenet.GoogLeNet,
        # 'googlenetbn': googlenetbn.GoogLeNetBN,
        # 'googlenetbn_fp16': googlenetbn.GoogLeNetBNFp16,
        # 'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
        # 'resnext50': resnext50.ResNeXt50,
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('image', help='input image path for predict')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='resnet50',
                        help='Convnet architecture')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--trainersnapshot', '-s', default='snapshot_epoch-100',
                        help='Trainer snapshot file (computed by train_imagenet.py)')
    parser.add_argument('--top', '-t', default=3,
                        help='Number of top score')
    args = parser.parse_args()

    # Initialize the model to train
    model = archs[args.arch]()
    chainer.serializers.load_npz(args.trainedmodel, model, path='updater/model:main/')
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
        model.to_gpu()
    # Load the mean file
    mean = np.load(args.mean).astype(np.float32)

    # Load the image #1
    # image = np.asarray(Image.open(args.image))
    # Load the image #2
    image = load_crop_image(args.image)
    cv_image_rgb = image[:, :, ::-1].copy()
    image = np.asarray(Image.fromarray(cv_image_rgb))
    image = image.transpose(2, 0, 1).astype(np.float32)

    # Crop the image
    crop_size = model.insize
    _, h, w = image.shape
    random_flag = True
    if random_flag:
        # Randomly crop a region and flip the image
        top = random.randint(0, h - crop_size - 1)
        left = random.randint(0, w - crop_size - 1)
        if random.randint(0, 1):
            image = image[:, :, ::-1]
    else:
        # Crop the center
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
    bottom = top + crop_size
    right = left + crop_size
    image = image[:, top:bottom, left:right].astype(np.float32)
    image -= mean[:, top:bottom, left:right]
    image *= (1.0 / 255.0)  # Scale to [0, 1]

    # Predict the image
    image = model.xp.asarray(image[None, ...])
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = predict_resnet50(model, image)
    y = to_cpu(y.array)

    categories = np.loadtxt("labels.txt", str, delimiter=",")
    # Print label number #1
    # pred_label = y.argmax(axis=1)
    # print('#1 ', categories[pred_label[0]])
    # Print label #2
    top = 3
    prediction = sorted(zip(y[0].tolist(), categories),reverse=True)
    for rank, (score, name) in enumerate(prediction[:top], start=1):
        print('#{} | {} | {}'.format(rank, name, score * 100))

if __name__ == '__main__':
    main()
