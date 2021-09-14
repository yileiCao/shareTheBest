import os

import cv2
from torch.utils.data import Dataset, DataLoader
from PyQt_function.nima_model import Nimanet
from PyQt_function.lut_model import *
from torchvision import transforms
from PIL import Image


def set_up_nima():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # model_state = "/Users/yileicao/Documents/Birmingham_project/summer_project/model_para_test/mnv3-pad-small_epoch60.pth"
    model_state = "/Users/yileicao/Documents/Birmingham_project/summer_project/model_para_test/mnv3_best_64.pth"
    model = Nimanet("mobileNetV3Small")
    state = torch.load(model_state, map_location=torch.device('cpu'))
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def set_up_lut():
    use_cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    path_to_model_state = "/Users/yileicao/Documents/Birmingham_project/summer_project/model_parameter"
    LUT0 = Generator3DLUT_zero()
    LUT1 = Generator3DLUT_zero()
    LUT2 = Generator3DLUT_zero()
    # LUT3 = Generator3DLUT_zero()
    # LUT4 = Generator3DLUT_zero()
    classifier = Classifier()
    trilinear = TrilinearInterpolation()

    if use_cuda:
        LUT0 = LUT0.cuda()
        LUT1 = LUT1.cuda()
        LUT2 = LUT2.cuda()
        # LUT3 = LUT3.cuda()
        # LUT4 = LUT4.cuda()
        classifier = classifier.cuda()

    # Load pretrained models
    LUTs = torch.load(os.path.join(path_to_model_state, "LUTs.pth"), map_location=torch.device('cpu'))
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    # LUT3.load_state_dict(LUTs["3"])
    # LUT4.load_state_dict(LUTs["4"])
    LUT0.eval()
    LUT1.eval()
    LUT2.eval()
    # LUT3.eval()
    # LUT4.eval()
    classifier.load_state_dict(torch.load(
        os.path.join(path_to_model_state, "classifier.pth"), map_location=torch.device('cpu')))
    classifier.eval()
    return LUT0, LUT1, LUT2, classifier, trilinear


def enhancement_result(raw_result, lut0, lut1, lut2, classifier, trilinear, nima):
    enhanced_images = []
    for index in range(len(raw_result)):
        image = raw_result[index][0]
        enhanced_image = image_enhancement(image, lut0, lut1, lut2, classifier, trilinear)
        enhanced_images.append(enhanced_image)
        if not index % 50:
            print(f"{index} images has been enhanced")
    enhanced_dataset = MyDataset(enhanced_images)
    enhanced_dataloader = DataLoader(enhanced_dataset, batch_size=32, shuffle=False)
    mean, std = evaluate_frames(nima, enhanced_dataloader)
    result = list(zip(enhanced_images, mean, std))
    return result


def image_enhancement(image, lut0, lut1, lut2, classifier, trilinear):
    image = transforms.functional.to_tensor(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    lut = generate_LUT(image, lut0, lut1, lut2, classifier)
    _, result = trilinear(lut, image)
    ndarr = result.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu',torch.uint8).numpy()
    return ndarr


def generate_LUT(img, LUT0, LUT1, LUT2, classifier):
    pred = classifier(img).squeeze()
    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT
    return LUT

def video_result(model, filePath):
    frames = open_video(filePath)
    # frames = open_file(filePath)
    dataset = MyDataset(frames)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean, std = evaluate_frames(model, dataloader)
    result = list(zip(frames, mean, std))
    return result

def images_result(model, filePath):
    # frames = open_video(filePath)
    frames = open_file(filePath)
    dataset = MyDataset(frames)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean, std = evaluate_frames(model, dataloader)
    result = list(zip(frames, mean, std))
    return result

def frame_eva(model, image):
    transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    x = Image.fromarray(image.astype(np.uint8))
    x = transform(x)
    x = torch.unsqueeze(x, 0)
    mean, std = model(x)
    return image, float(mean), float(std)

def frame_enhance(raw_image, lut0, lut1, lut2, classifier, trilinear, nima):
    image = raw_image
    enhanced_image = image_enhancement(image, lut0, lut1, lut2, classifier, trilinear)
    return frame_eva(nima, enhanced_image)



def open_video(filePath):

    cap = cv2.VideoCapture(str(filePath))
    assert (cap.isOpened())

    # init empty output frames (N x H x W x C)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read video
    # frame = np.empty((max_frames, height, width, 3), np.dtype('float32'))
    frames = []
    fc = 0
    ret = True
    while fc < max_frames and ret:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = Image.fromarray(frame, 'RGB')
        # change here
        # frame = frame[:, 280:1000]
        frames.append(frame)
        fc += 1
    cap.release()
    return frames


def open_file(filePath):
    items = os.listdir(filePath)
    images = []
    for item in items:
        if item.endswith(".jpg"):
            load_path = os.path.join(filePath, item)
            image = Image.open(load_path).convert("RGB")
            images.append(np.array(image))
    return images


def evaluate_frames(model, data):
    torch.set_grad_enabled(False)
    means = []
    stds = []
    for batch, X in enumerate(data):
        mean, std = model(X)
        means.extend(mean.numpy().tolist())
        stds.extend(std.numpy().tolist())
        print(f"Batch{batch} finished")
    return means, stds


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return transforms.functional.pad(image, padding, 0, 'constant')

class MyDataset(Dataset):
    def __init__(self, data, transform_type="nima"):
        self.data = data
        self.transform_type = transform_type
        self.transform = transforms.Compose(
            [
                SquarePad(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def __getitem__(self, index):
        x = self.data[index]
        x = Image.fromarray(self.data[index].astype(np.uint8))
        x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)


def main():
    video_result("/Users/yileicao/Documents/Birmingham_project/summer_project/ppt")


if __name__ == "__main__":
    main()

