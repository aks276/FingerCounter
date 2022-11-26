import cv2
import torch

import os

from data import test_transform

from torchvision.models import vgg16

model = vgg16()
model.load_state_dict(torch.load('./models/model_19.pth', map_location='cpu'))

# vid = cv2.VideoCapture(0)

# while(True):
#     success, frame = vid.read()
#     cv2.imshow('frame', frame)

#     frame = test_transform(image=frame)["image"]
#     print(torch.argmax(model(frame.unsqueeze(0))))

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# vid.release()

# cv2.destroyAllWindows()

for img in os.listdir('./imgs'):
    image = cv2.imread(os.path.join('./imgs', img))
    image = test_transform(image=image)["image"]
    image = image.unsqueeze(0)
    print(torch.argmax(model(image)))