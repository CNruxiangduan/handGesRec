from ultralytics import YOLO
import torch
import cv2
import numpy as np
# select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set threshold
threshold = 0.5

# load a pretrained model (recommended for best training results)
model = YOLO('./weighs/det/best.pt')    
clsModel = YOLO("./weights/cls/best.pt")            
model.to(device)
clsModel.to(device)

cap = cv2.VideoCapture(0)
while(True):
    # predict on an image
    ret,img = cap.read()
    
    if ret:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inputs = [img_rgb ,img_rgb]
        results = model.predict(img)
        # print(results)
        
        boxes = results[0].boxes
        if len(boxes)>0:
            for box in boxes:
                # box = boxes[0]
                
                
                box = box.xyxy.cpu().numpy()
                # print(box)
                
                for res in box:
                    
                    # print(box.shape)
                    x0 = int(res[0])
                    y0 = int(res[1])
                    x1 = int(res[2])
                    y1 = int(res[3])
                    roi = img_rgb[y0:y1, x0:x1]
                    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite('1.jpg',roi)

                    inputs = [roi ,roi]
                    clsOut = clsModel.predict(inputs)
                    # print("##################################33\n")
                    # print(clsOut[0].probs)
                    out_probs = clsOut[0].probs.cpu().numpy()
                    index = np.argmax(out_probs)
                    score = out_probs[index]
                    pred = clsOut[0].names[index]
                    cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),1)

                    font1 = cv2.FONT_HERSHEY_SIMPLEX
                    # font scale for the text being specified
                    fontScale1 = 1
                    # Blue color for the text being specified from BGR
                    color1 = (0, 255, 0)
                    # Line thickness for the text being specified at 2 px
                    thickness1 = 2

                    cv2.putText(img,pred+":"+str(score),(x0,y0-10),font1, fontScale1, color1, thickness1, cv2.LINE_AA)
                    # print(clsOut[0].names[index])
                    # print("##################################33\n")
    cv2.imshow("image", img)
    key=cv2.waitKey(1)
    if key==27:
        break