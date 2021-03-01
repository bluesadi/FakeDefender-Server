import os
import time

from cv2 import cv2
import numpy as np
from collections import defaultdict

from PIL import Image
from torch import torch
import torch.nn.functional as F
from torchvision import transforms as T

from face_utils import norm_crop, FaceDetector
from model_def import WSDAN, xception

from face_marker import FaceMarker


class DFDCLoader:
    def __init__(self, video_dir, face_detector, transform=None,
                 batch_size=25, frame_skip=9, face_limit=25):
        self.video_dir = video_dir
        self.file_list = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))

        self.transform = transform
        self.face_detector = face_detector

        self.batch_size = batch_size
        self.frame_skip = frame_skip
        self.face_limit = face_limit

        self.record = defaultdict(list)
        self.score = defaultdict(lambda: 0.5) #最终的预测概率
        self.feedback_queue = []
        self.boxes = []

    def iter_one_face(self):
        for fname in self.file_list:
            path = os.path.join(self.video_dir, fname)
            reader = cv2.VideoCapture(path)
            face_count = 0

            while True:
                for _ in range(self.frame_skip):
                    reader.grab()

                success, img = reader.read()
                # 检测这一帧是否存在
                if not success:
                    break

                boxes, landms = self.face_detector.detect(img)
                self.boxes = boxes
                if boxes.shape[0] == 0:
                    continue

                for box in boxes:
                    faceX = int(box[0].item())
                    faceY = int(box[1].item())
                    faceW = int(box[2].item()) - faceX
                    faceH = int(box[3].item()) - faceY
                    marker = FaceMarker(img, faceX, faceY, faceW, faceH, 0.6)
                    marker.mark()
                cv2.imshow("hhh", img)
                cv2.waitKey()

                areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                order = areas.argmax()
                boxes = boxes[order]
                landms = landms[order]

                # Crop faces
                landmarks = landms.numpy().reshape(5, 2).astype(np.int)
                img = norm_crop(img, landmarks, image_size=320)
                aligned = Image.fromarray(img[:, :, ::-1])

                if self.transform:
                    aligned = self.transform(aligned)

                yield fname, aligned

                # Early stop
                face_count += 1
                if face_count == self.face_limit:
                    break

            reader.release()

    def __iter__(self):
        self.record.clear()
        self.feedback_queue.clear()

        batch_buf = []
        t0 = time.time()
        batch_count = 0

        for fname, face in self.iter_one_face():
            self.feedback_queue.append(fname)
            batch_buf.append(face)

            if len(batch_buf) == self.batch_size:
                yield torch.stack(batch_buf)

                batch_count += 1
                batch_buf.clear()

                if batch_count % 10 == 0:
                    elapsed = 1000 * (time.time() - t0)
                    print("T: %.2f ms / batch" % (elapsed / batch_count))

        if len(batch_buf) > 0:
            yield torch.stack(batch_buf)

    def feedback(self, pred):
        accessed = set()

        for score in pred:
            fname = self.feedback_queue.pop(0)
            accessed.add(fname)
            self.record[fname].append(score)

        for fname in sorted(accessed):
            self.score[fname] = np.mean(self.record[fname])
            print("[%s] %.6f" % (fname, self.score[fname]))


def main():
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    test_dir = "./input/test_videos"

    face_detector = FaceDetector()
    face_detector.load_checkpoint("./input/dfdc-pretrained-2/RetinaFace-Resnet50-fixed.pth")
    loader = DFDCLoader(test_dir, face_detector, T.ToTensor())

    model1 = xception(num_classes=2, pretrained=False)
    ckpt = torch.load("./input/dfdc-pretrained-2/xception-hg-2.pth", map_location=torch.device('cpu'))
    model1.load_state_dict(ckpt["state_dict"])
    model1 = model1.cpu()
    model1.eval()

    model2 = WSDAN(num_classes=2, M=8, net="xception", pretrained=False).cpu()
    ckpt = torch.load("./input/dfdc-pretrained-2/ckpt_x.pth", map_location=torch.device('cpu'))
    model2.load_state_dict(ckpt["state_dict"])
    model2.eval()

    model3 = WSDAN(num_classes=2, M=8, net="efficientnet", pretrained=False).cpu()
    ckpt = torch.load("./input/dfdc-pretrained-2/ckpt_e.pth", map_location=torch.device('cpu'))
    model3.load_state_dict(ckpt["state_dict"])
    model3.eval()

    zhq_nm_avg = torch.Tensor([.4479, .3744, .3473]).view(1, 3, 1, 1).cpu()
    zhq_nm_std = torch.Tensor([.2537, .2502, .2424]).view(1, 3, 1, 1).cpu()

    for batch in loader:
        batch = batch.cpu()

        i1 = F.interpolate(batch, size=299, mode="bilinear")
        i1.sub_(0.5).mul_(2.0)
        o1 = model1(i1).softmax(-1)[:, 1].cpu().numpy()

        i2 = (batch - zhq_nm_avg) / zhq_nm_std
        o2, _, _ = model2(i2)
        o2 = o2.softmax(-1)[:, 1].cpu().numpy()

        i3 = F.interpolate(i2, size=300, mode="bilinear")
        o3, _, _ = model3(i3)
        o3 = o3.softmax(-1)[:, 1].cpu().numpy()

        out = 0.2 * o1 + 0.7 * o2 + 0.1 * o3
        loader.feedback(out)

        print(loader.record)
        print(loader.score)


if __name__ == "__main__":
    main()
