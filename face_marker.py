import cv2


class FaceMarker:

    def __init__(self, img, faceX, faceY, faceW, faceH, fakeProb):
        self.img = img
        self.faceX = faceX
        self.faceY = faceY
        self.faceW = faceW
        self.faceH = faceH
        self.fakeProb = fakeProb
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_GREEN = (0, 255, 0)

    def mark(self):
        color = self.COLOR_GREEN if self.fakeProb < 0.5 else self.COLOR_RED
        facePtr1 = (self.faceX, self.faceY)
        facePtr2 = (self.faceX + self.faceW, self.faceY + self.faceH)
        cv2.rectangle(self.img, facePtr1, facePtr2, color, 2)
        cv2.putText(self.img, f'Fake: {self.fakeProb}', (facePtr1[0], facePtr2[1] + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, color, 2)
