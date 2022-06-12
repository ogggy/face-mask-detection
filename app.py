from flask import Flask, render_template, Response, url_for, redirect, flash, request
import urllib.request
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from os.path import join, dirname, realpath

# Define a flask app
app = Flask(__name__)
wsgi_app = app.wsgi_app

#LoadModel and Config
BASE_PATH = join(dirname(realpath(__file__)))
PROTOTXT_PATH = join(BASE_PATH, 'face-detector-config/deploy.prototxt')
WEIGHT_PATH = join(BASE_PATH, 'face-detector-config/res10_300x300_ssd_iter_140000.caffemodel')
MODEL_PATH = join(BASE_PATH, 'mask-model/model.h5')
MODEL = load_model(MODEL_PATH)

#Config
UPLOAD_FOLDER = '/static/uploads'
app.secret_key = "secrect key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS_IMG = set(['png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS_VIDEO = set(['mp4'])
# vs = cv2.VideoCapture(0)

#-------------------------#Router ----------------------------------------------------------
@app.route('/')
def index():
    delete_all_file_in_uploads()
    return render_template('index.html')
    
@app.route('/live')
def live_stream():
    return render_template('live.html')
    # return Response(live(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream')
def stream():
    return Response(live(PROTOTXT_PATH, WEIGHT_PATH, MODEL), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Không có ảnh nào được tải lên')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        UPLOADS_PATH = join(dirname(realpath(__file__)), 'static/uploads/', file.filename)
        file.save(os.path.join(UPLOADS_PATH))
        detect_image(PROTOTXT_PATH, WEIGHT_PATH, MODEL,UPLOADS_PATH)
        # file.save(os.path.join(UPLOADS_PATH))
        flash('Tải ảnh thành công', 'success') #'Tải ảnh thành công'
        
        # return render_template('index.html', filename=filename)
        # flash(str(url_for(filename)))
        return render_template('index.html', filename=filename)
    else:
        flash('Chỉ những định dạng .png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code = 301)

# @app.route('/video')
# def video():
#     return render_template('video.html')


# @app.route('/video', methods=['POST'])
# def upload_video():
# 	if 'file' not in request.files:
# 		flash('No file part')
# 		return redirect(request.url)
# 	file = request.files['file']
# 	if file.filename == '':
# 		flash('No video selected for uploading')
# 		return redirect(request.url)
# 	else:
# 		filename = secure_filename(file.filename)
# 		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
# 		#print('upload_video filename: ' + filename)
# 		flash('Video successfully uploaded and displayed below')
# 		return render_template('video.html', filename=filename)

# @app.route('/display/<filename>')
# def display_video(filename):
	#print('display_video filename: ' + filename)
	# return redirect(url_for('static', filename='uploads/' + filename), code=301)

#-------------------#function needed-----------------------------------------------------------
def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it	
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # convert from BGR to RGB
            # ordering, resize it to 224x224 and preprocess 
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

def live(prototxtPath, weightsPath, model):

    # pass the blob through the network and obtain the face detections
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = model

    vs = cv2.VideoCapture(0)
    time.sleep(2.0)
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        success, frame = vs.read()
        if not success:
            break
        # frame = imutils.resize(frame, width=600)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # class label and color 
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        # show the output frame
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF

        # # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        # 	break


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMG

def detect_image(prototxtPath, weightsPath, model, img_path):
    #Load Model
   
    # ###args paths
    # prototxtPath = '/Users/alann/Desktop/iuh/CNM_final/Report/face_detector/deploy.prototxt'
    # weightsPath = '/Users/alann/Desktop/iuh/CNM_final/Report/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
    # model_path = '/Users/alann/Desktop/iuh/CNM_final/Report/model1/model.h5'
    # model_predict = model

    #Read Image
    image = cv2.imread(img_path)
    # orig = image.copy()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            # frame = cv2.imencode('.jpg', image)[1].tobytes()
            # yield (b'--frame\r\n'
            #     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            cv2.imwrite(img_path, image)

def delete_all_file_in_uploads():
    import os, shutil
    folder = join(BASE_PATH, 'static/uploads')
    if os.listdir(folder) == []:
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
             shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


#-----------------Main---------------------------------------------------------------------------
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    app.run(host='0.0.0.0', port=81, debug=True)
    # print(BASE_PATH)
    
