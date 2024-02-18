from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
emotion_counts = {emotion: 0 for emotion in EMOTIONS}

app = Flask(__name__)
app.secret_key = 'e7f59b56d8a4448fbd61946a14926f9b'  # Replace with your own secret key

camera = cv2.VideoCapture(0)

improvement_suggestions = {
    "happy": [
        "Try to smile and laugh more.",
        "Maintain eye contact to engage with your audience.",
        "Use positive language to keep the mood upbeat."
    ],
    "sad": [
        "Try to use a softer tone of voice.",
        "Use body language to express sadness.",
        "Share personal stories to connect with your audience."
    ],
    "angry": [
        "Use a firm tone of voice to express anger.",
        "Use body language to express anger.",
        "Use powerful language to convey your message."
    ],
    "neutral": [
        "Use hand gestures to emphasize your points.",
        "Maintain eye contact to engage with your audience.",
        "Use a variety of vocal tones to keep your audience engaged."
    ]
}


def gen_frames():
    while True:
        frame = camera.read()[1]
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            emotion_counts[label] += 1
        else:
            continue
        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', frameClone)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_canvas():
    while True:
        frame = camera.read()[1]
        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            emotion_counts[label] += 1
        else:
            continue
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', canvas)
        canvas = jpeg.tobytes()
        yield (b'--canvas\r\n' b'Content-Type: image/jpeg\r\n\r\n' + canvas + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/canvas_feed')
def canvas_feed():
    return Response(gen_canvas(), mimetype='multipart/x-mixed-replace; boundary=canvas')


@app.route('/emotion_stats')
def emotion_stats():
    print("Emotion counts: ", jsonify(emotion_counts))
    return jsonify(emotion_counts)


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    print("Session variables: ", session)  # Print the session variables
    selected_emotion = "happy"  # Define selected_emotion with a default value
    if 'emotion' in session:
        selected_emotion = session['emotion']  # Retrieve the selected emotion from the session
        print("Selected emotion: ", selected_emotion)
    total = sum(emotion_counts.values())
    feedback = ""
    if total > 0:
        max_emotion = max(emotion_counts, key=emotion_counts.get)
        max_percentage = (emotion_counts[max_emotion] / total) * 100
        print(f"{max_emotion}: {max_percentage}%")  # Print the emotion with the max percentage

        feedback = ""
        if selected_emotion == max_emotion and max_percentage > 50:
            feedback += f"Good job! You successfully conveyed a sense of {max_emotion} in your speech. "
        else:
            feedback += f"You seemed a bit too {max_emotion} at times. Remember, you're aiming for a {selected_emotion} demeanor. Try to express more {selected_emotion}. "
    else:
        feedback = "No facial expression detected."

    return jsonify({"feedback": feedback})



@app.route('/reset')
def reset():
    global emotion_counts
    emotion_counts = {emotion: 0 for emotion in EMOTIONS}
    return "Emotion counts reset"


@app.route('/', methods=['GET', 'POST'])
def landing():
    if request.method == 'POST':
        selected_emotion = request.form.get('emotion')
        session['emotion'] = selected_emotion  # Store the selected emotion in the session
        return redirect(url_for('index'))
    return render_template('landing.html')


@app.route('/index', methods=['GET', 'POST'])
def index():
    print("Request method: ", request.method)  # Print the request method
    print("Session variables: ", session)  # Print the session variables
    selected_emotion = "happy"  # Define selected_emotion with a default value
    if 'emotion' in session:
        selected_emotion = session['emotion']  # Retrieve the selected emotion from the session
        print("Selected emotion: ", selected_emotion)
    return render_template('index.html', improvements=improvement_suggestions.get(selected_emotion, []))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
