from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/mask_plate', methods=['POST'])
def mask_plate():
    try:
        # Get the image from the request
        img = request.files['image'].read()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
        print(f'Image shape: {img.shape}, type: {img.dtype}')

        # Load the classifier for detecting the vehicle’s number plate region
        plateCascade = cv2.CascadeClassifier('indian_license_plate.xml')

        # Set the scaleFactor and minNeighbors parameters
        scaleFactor = 1.3
        minNeighbors = 5

        # Detect the number plate and mark it with a green rectangle
        plateRect = plateCascade.detectMultiScale(img, scaleFactor, minNeighbors)
        # plateRect = plateCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
        print(f'Number of number plates detected: {len(plateRect)}')

        if len(plateRect) == 0:
            return jsonify({'error': 'No number plates detected in the image'})

        for (x,y,w,h) in plateRect:
            cv2.rectangle(img, (x+2,y), (x+w-3, y+h-5), (0,255,0), 3)

        # Mask the part other than the number plate
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        for (x,y,w,h) in plateRect:
            mask[y:y+h, x:x+w] = 0
        new_image = cv2.bitwise_and(img,img,mask=mask)
        
        # Encode the image and return it as a base64-encoded string
        _, img_encoded = cv2.imencode('.jpg', new_image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        return jsonify({'image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

