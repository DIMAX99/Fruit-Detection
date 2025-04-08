from flask import Flask, jsonify,request,send_file
from flask_cors import CORS
from PIL import Image
import io
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import base64

app=Flask(__name__)
CORS(app)
model=YOLO("../train5/weights/best.pt")

@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400
    
    file = request.files['file']
    image_pil = Image.open(file.stream).convert("RGB")
    image_np = np.array(image_pil)  # Convert to NumPy array for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = model(image_pil,conf=0.5)
    boxes = results[0].boxes
    class_name=[]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates   
        cls_id = int(box.cls[0].item())  # Convert tensor to int
        cls_name = model.names[cls_id]
        conf = float(box.conf[0].item())  # Get class name from id
        class_name.append({'name':cls_name,'confidence':conf})
        # Draw rectangle with thin line
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)


    # Convert back to PIL
    result_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_image)

    # Send result
    image_io = io.BytesIO()
    result_pil.save(image_io, format='PNG')
    image_io.seek(0)
    img_base=base64.b64encode(image_io.read()).decode('utf-8')

    return jsonify({
        'classes_detected':class_name,
        'image':img_base
    })
@app.route("/api/home",methods=['GET'])
def return_home():
    return jsonify({
        'message':"Hello World!"
    })

if __name__=='__main__':
    app.run(debug=True,port=8080)