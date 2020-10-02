
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./face-detection-pytorch')
from detectors import DSFD
import cv2
import torch
import os
from sklearn import preprocessing
import time
import datetime
import sys
import json

def load_cuda():
    if torch.cuda.is_available():
        print('CUDA is available.')
        print('GPU count : ', torch.cuda.device_count())
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('CUDA is not available.')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device {}\n'.format(device))
    return device


def create_label_dict(label_list):
    le = preprocessing.LabelEncoder()
    le.fit_transform(label_list)

    class_dict = {label: value for label, value in zip(le.classes_, le.transform(le.classes_))}
    class_inv_dict = {j: i for i, j in class_dict.items()}
    return class_dict, class_inv_dict


def conv_3x3(in_c, out_c, stride):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_c)
    )
    return conv


def conv_point(in_c, out_c, stride):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(out_c)
    )
    return conv


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        if in_c != out_c:
            # ダウンサンプリングが発生するブロック
            self.downsample = conv_point(in_c, out_c, stride=2)
            self.conv1 = conv_3x3(in_c, out_c, stride=2)
        else:
            self.downsample = None
            self.conv1 = conv_3x3(in_c, out_c, stride=1)

        self.conv2 = conv_3x3(out_c, out_c, stride=1)

    def forward(self, x):
        identity_x = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        # 恒等写像を用いたshortcut connection
        if self.downsample != None:
            identity_x = self.downsample(identity_x)
        x += identity_x
        x = F.relu(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.first_layer = BasicBlock(in_c, out_c)
        self.second_layer = BasicBlock(out_c, out_c)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.second_layer(x)
        return x

class MaskNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.conv2_x = ResidualBlock(64, 64)
        self.conv3_x = ResidualBlock(64, 128)
        self.conv4_x = ResidualBlock(128, 256)
        self.fc1 = nn.Linear(8 * 8 * 256, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = x.view(-1, 8 * 8 * 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_models(device, label_list, mask_path):
    dsfd_model = DSFD(device=device)

    # modified by Y.Ishizuka
    #model = MaskNet(label_list).cuda() if torch.cuda.is_available else MaskNet(label_list)
    model = MaskNet().cuda() if torch.cuda.is_available() else MaskNet()
    # end of modified
    model.load_state_dict(torch.load(mask_path))
    model = model.eval()
    return dsfd_model, model


def crop_face(frame, box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    box = [int(box[0] - width / 2), int(box[1] - height / 2), int(box[2] + width / 2), int(box[3] + height / 2)]
    for i in range(len(box)):
        if box[i] < 0:
            box[i] = 0
    frame_crop = frame[box[1]: box[3], box[0]: box[2]]
    return frame_crop

def detect_mask(frame_crop, transfor, class_inv_dict, model, device):
    box_tensor = transfor(cv2.resize(frame_crop, (64, 64)))
    output = model(box_tensor.unsqueeze(0).to(device))
    output = F.softmax(output, dim=1)
    print('   Output : mask - {:.3f},   no_mask - {:.3f}'.format(output[0, 0], output[0, 1]))
    class_num = int(output.argmax(dim=1, keepdim=True)[0, 0])
    class_name = class_inv_dict[class_num]
    class_acc = output[0, class_num]
    return class_name, class_acc

def main():
    args_list = sys.argv
    if len(args_list) == 4:
        file_dir = args_list[1]
        out_path = args_list[2]
        out_flag = args_list[3]
    elif len(args_list) == 3:
        file_dir = args_list[1]
        out_path = args_list[2]
        out_flag = False
    elif len(args_list) == 2:
        if os.path.splitext(args_list[1])[1] == '.json':
            file_dir = './sample_dir'
            out_path = args_list[1]
            out_flag = False
        else:
            file_dir = args_list[1]
            out_path = 'sample.json'
            out_flag = False
    else:
        file_dir = './sample_dir'
        out_path = './sample.json'
        out_flag = False

    device = load_cuda()
    label_list = ['mask', 'no-mask']
    class_dict, class_inv_dict = create_label_dict(label_list)

    dsfd_model, model = load_models(device, label_list, mask_path='./weights/mask_detect_resnet-0626_8x8.pth')

    trans = torchvision.transforms.ToTensor()
    norm = torchvision.transforms.Normalize((0.5, ), (0.5, ))
    transfor = torchvision.transforms.Compose([trans, norm])

    detect_thre = 0.9   # 顔検知の閾値、閾値未満は棄却
    #count_thre = 0.95  # カウントの閾値、
    count_thre = 0.95   # mask detectの出力はsoftmaxに通していないため、値は割合でではないことに注意
    sampling_T = 1      # フォルダ内のファイルをどの程度の周期で読み込むか
    print('サンプリング周期：', sampling_T, '\n')

    file_list = os.listdir(file_dir)
    pict_size = cv2.imread(file_dir + '/' + file_list[0]).shape
    #crop_box = [0, int(pict_size[0]/2), int(pict_size[1]), int(pict_size[0])]

    count_dict = {class_name: 0 for class_name in [k for k in class_dict.keys()] + ['others']}
    start = time.time()

    cc = 0

    for frame_num, file_name in enumerate(file_list, 1):
        print(frame_num)
        # 全ファイルで検知を行うなら不要、一応残してある分岐
        if frame_num % sampling_T == 0:
            frame = cv2.imread(file_dir + '/' + file_name)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print('Now processing : {}/{} -> {:.2f}%'.format(frame_num, len(file_list), frame_num / len(file_list) * 100))

            #frame = frame[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]]
            boxes = dsfd_model.detect_faces(frame, conf_th=detect_thre, scales=[0.5, 1])
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cc += len(boxes)

            if boxes is not None:
                for box in boxes:
                    # 閾値は検出プロセスの一環に組み込まれているため分岐は不要
                    # if box[4] < detect_thre:
                    #    continue
                    frame_crop = crop_face(frame, box).copy()
                    class_name, class_acc = detect_mask(frame_crop, transfor, class_inv_dict, model, device)

                    if class_acc < count_thre:
                        class_name = 'others'
                    count_dict[class_name] += 1

                    if out_flag=='True':
                        width = box[2] - box[0]
                        height = box[3] - box[1]
                        if class_acc >= count_thre:
                            print('      ' + class_name)
                            if class_name == "mask":
                                cv2.rectangle(frame, (int(box[0] - width / 2), int(box[1] - height / 2)),
                                              (int(box[2] + width / 2), int(box[3] + height / 2)), (255, 0, 0), 5)
                            if class_name == 'no-mask':
                                cv2.rectangle(frame, (int(box[0] - width / 2), int(box[1] - height / 2)),
                                              (int(box[2] + width / 2), int(box[3] + height / 2)), (0, 0, 255), 5)
                        else:
                            cv2.rectangle(frame, (int(box[0] - width / 2), int(box[1] - height / 2)),
                                          (int(box[2] + width / 2), int(box[3] + height / 2)), (0, 255, 0), 5)
                        # 検証用に検出画像を保存している、不要なら外して問題ない
                        cv2.imwrite('./output/' + class_name + '/' + str(count_dict[class_name]) + '.jpg', frame_crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if out_flag=='True':
                cv2.imwrite('./output/full_size/' + str(frame_num) + '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

    print('\nFinished.')
    print('Processing time : {:.2f}'.format((time.time() - start)/60))
    print(count_dict)
    print(cc)

    if os.path.isfile(out_path):
        with open(out_path) as f:
            out_dict = json.load(f)
        out_dict[str(datetime.datetime.now())] = {'Target': file_dir, 'result': str(count_dict)}
        with open(out_path, 'w') as f:
            json.dump(out_dict, f)
    else:
        out_dict = {str(datetime.datetime.now()): {'Target': file_dir, 'result': str(count_dict)}}
        with open(out_path, 'w') as f:
            json.dump(out_dict, f)
    '''
    with open('./result.txt', mode='a') as f:
        f.write('\n\n')
        f.write('Date: ' + str(datetime.datetime.now()) + '\n')
        f.write('Tartget directory: ' + file_dir + '\n')
        f.write(str(count_dict))
    '''

if __name__ == '__main__':
    main()
