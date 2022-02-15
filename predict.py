import argparse
import torch
import torch.nn.functional as F

import glob
import cv2
import numpy as np
import os

from model import LinearNet

def load_image(img_path):
    img = cv2.imread(img_path)
    resize = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray)  # background of image in train is black color
    data = invert/255 # normailize 
    data = data.reshape(1, 1, 28, 28).astype(np.float32)
    return data

def predict(model, tensor_image):
    output = model(tensor_image)
    output = F.softmax(output, dim=1)
    # print(output)
    pred = output.argmax(dim=1, keepdim=True)
    label = pred.item()
    prob = output[0][label].item()
    return label, prob

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST predict Example')
    parser.add_argument('--eval', type=str, required=True, metavar='N',
                        help='eval folder path')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model', type=str, required=True, metavar='N',
                        help='eval folder path')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    image_files = glob.glob(args.eval + '*.png')

    model = LinearNet().to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    correct = 0
    for image in image_files:
        filename = os.path.basename(image)
        actual_label = os.path.splitext(filename)[0]
        data = load_image(image)

        tensor = torch.from_numpy(data) # convert to tensor image
        # plt.imshow(  tensor.permute(1, 2, 0)  )   # show tensor image (RGB)
        tensor_ = np.squeeze(tensor)    #show tensor image (gray)
        # plt.imshow( tensor_ )

        if not args.no_cuda:
            tensor = tensor.to(device)      # run with GPU

        label, prob = predict(model, tensor)

        if label == int(actual_label):
            correct += 1
        print(f'file: {image}\nPredicted label: {label}, prob: {prob}')

    correct_rate = correct / len(image_files) * 100
    print(f'Model: {args.model}, Total correct rate: {correct_rate}%')

if __name__ == '__main__':
    main()