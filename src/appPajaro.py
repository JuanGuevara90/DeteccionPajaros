import time
import threading
from flask import Response, render_template, Flask
import cv2
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.utils.misc import Timer
import sys
import sendRealDB 
import serial


port = serial.Serial("/dev/ttyACM0", baudrate=9600)
if len(sys.argv) < 5:
    #print('Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
threshold = sys.argv[4]



class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-large-ssd-lite':
    net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb3-small-ssd-lite':
    net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    #print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite' or net_type == "mb3-large-ssd-lite" or net_type == "mb3-small-ssd-lite":
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    #print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)



# Image frame sent to the Flask object
global video_frame
video_frame = None

# Use locks for thread-safe viewing of frames in multiple browsers
global thread_lock 
thread_lock = threading.Lock()

# Create the Flask object for the application
app = Flask(__name__)

def captureFrames():
    global video_frame, thread_lock

    # Video capturing from OpenCV
    video_capture = cv2.VideoCapture(0)

    while True and video_capture.isOpened():
        try:
            return_key, frame = video_capture.read()
            if not return_key:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, labels, probs = predictor.predict(image, 10, 0.4)
            #print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
            for i in range(boxes.size(0)):
                label = f"{class_names[labels[i]]}:{probs[i]:.2f}"
                split_val=label.split(sep=":",maxsplit=2)
                prob =float(split_val[1])
                if(class_names[labels[i]]=='Bird' and prob>=float(threshold)):
                    print("Guardar")
                    cv2.imwrite('imagen.jpeg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    sendRealDB.sendDataFireBase('imagen.jpeg')
                    #port.write(b"B\n")
                    

            # Create a copy of the frame and store it in the global variable,
            # with thread safe access
            with thread_lock:
                video_frame = frame.copy()
            
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break

        except Exception as e:
            print("Aqui")
            print(e)

    video_capture.release()
        
def encodeFrame():
    global thread_lock
    while True:
        try:
            # Acquire thread_lock to access the global video_frame object
            with thread_lock:
                global video_frame
                if video_frame is None:
                    continue
                return_key, encoded_image = cv2.imencode(".jpg", video_frame)
                if not return_key:
                    continue

            # Output image as a byte array
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encoded_image) + b'\r\n')
        except Exception as e:
            print("Aqui")
            print(e)

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(encodeFrame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results')
def results():
    """Video streaming results page."""
    return render_template('results.html')

@app.route('/js')
def js():
    """Video streaming javascript page."""
    return render_template('index.js')


# check to see if this is the main thread of execution
if __name__ == '__main__':

    # Create a thread and attach the method that captures the image frames, to it
    process_thread = threading.Thread(target=captureFrames)
    process_thread.daemon = True

    # Start the thread
    process_thread.start()

    # start the Flask Web Application
    # While it can be run on any feasible IP, IP = 0.0.0.0 renders the web app on
    # the host machine's localhost and is discoverable by other machines on the same network 
    app.run("0.0.0.0", port="8000")