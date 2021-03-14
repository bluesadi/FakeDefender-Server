from cv2 import cv2
import numpy as np
from PIL import Image
from torch import torch
import torch.nn.functional as F
from torchvision import transforms as T
from face_utils import norm_crop, FaceDetector
from model_def import WSDAN, xception
from flask import Flask, jsonify, request
import json
import base64

app = Flask(__name__)


class DFDCImageLoader:
    def __init__(self, face_detector, transform=None):
        self.face_detector = face_detector
        self.transform = transform
        self.zhq_nm_avg = torch.Tensor([.4479, .3744, .3473]).view(1, 3, 1, 1).cpu()
        self.zhq_nm_std = torch.Tensor([.2537, .2502, .2424]).view(1, 3, 1, 1).cpu()

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

        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def filterFaces(self, faces):
        ind = []
        for i in range(len(faces)):
            face = faces[i]
            repeated = False
            for out_i in ind:
                out_face = faces[out_i]
                if abs(face[0].item() - out_face[0].item()) < 10 and abs(
                        face[1].item() - out_face[1].item()) < 10 and abs(face[2].item() - out_face[2].item()) < 10:
                    repeated = True
                    break
            if not repeated:
                ind.append(i)
        return ind

    def predictOneFace(self, img, landmarks):
        model1 = self.model1
        model2 = self.model2
        model3 = self.model3

        img = norm_crop(img, landmarks, image_size=320)
        aligned = Image.fromarray(img[:, :, ::-1])

        if self.transform:
            aligned = self.transform(aligned)

        batch_buf = []
        batch_buf.append(aligned)
        batch = torch.stack(batch_buf)
        batch = batch.cpu()

        i1 = F.interpolate(batch, size=299, mode="bilinear")
        i1.sub_(0.5).mul_(2.0)
        o1 = model1(i1).softmax(-1)[:, 1].cpu().detach().numpy()

        i2 = (batch - self.zhq_nm_avg) / self.zhq_nm_std
        o2, _, _ = model2(i2)
        o2 = o2.softmax(-1)[:, 1].cpu().detach().numpy()

        i3 = F.interpolate(i2, size=300, mode="bilinear")
        o3, _, _ = model3(i3)
        o3 = o3.softmax(-1)[:, 1].cpu().detach().numpy()

        out = 0.2 * o1 + 0.7 * o2 + 0.1 * o3
        return out[0]

    def predict(self, img):
        boxes, landms = self.face_detector.detect(img)
        if boxes.shape[0] == 0:
            return 0.0
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # order = areas.argmax()
        ind = self.filterFaces(boxes)
        boxes = boxes[ind]
        landms = landms[ind]
        # boxes = boxes[order]
        # landms = landms[order]
        scores = []
        for landmark in landms:
            landmarks = landmark.detach().numpy().reshape(5, 2).astype(np.int)
            scores.append(self.predictOneFace(img, landmarks))

        return boxes, scores

# load model
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
face_detector = FaceDetector()
face_detector.load_checkpoint("./input/dfdc-pretrained-2/RetinaFace-Resnet50-fixed.pth")
loader = DFDCImageLoader(face_detector, T.ToTensor())
print("loader ok")

@app.route('/')
def hello():
    return "Welcome to Deepfake Server."


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        data = json.loads(data, strict=False)
        uuid = data["uuid"]
        img_encode = base64.b64decode(data["image"])
        img = cv2.imdecode(np.frombuffer(img_encode, np.uint8), cv2.IMREAD_COLOR)
        faces, scores = loader.predict(img)

        response = {
            "uuid": uuid,
            "faceNum": len(faces)
        }
        if len(faces) != 0:
            faceList = []
            for i in range(len(faces)):
                face = faces[i]
                faceList.append({
                    "x1": int(face[0].item()),
                    "y1": int(face[1].item()),
                    "x2": int(face[0].item()),
                    "y2": int(face[0].item()),
                    "score": scores[i].item()
                })
            response["faces"] = faceList

        print(json.dumps(response))
        print("The server is still running...")
        return jsonify(response)


if __name__ == "__main__":
    app.run()
