from tkinter import *
from PIL import ImageTk, Image, ImageGrab
import numpy as np
from tkinter import ttk
import time
import os
from os.path import join
import cv2
from datetime import datetime

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print("ROOT_DIR", ROOT_DIR)

# Get usage
import psutil

pid = os.getpid()
python_process = psutil.Process(pid)

# GPU_usage = []
GPU_memory = []
CPU_usage = []
RAM_usage = []

GPU = False

# Turn on/off GPU
if not GPU:
    tf.config.set_visible_devices([], 'GPU')

# Global variables
ts = datetime.now().strftime("%m%d%Y_%H%M%S")
DST_PATH = join(ROOT_DIR, "RESULTADOS", ts)
try:
    os.makedirs(DST_PATH)    
    print("Directory " , DST_PATH ,  " Created ")
except FileExistsError:
    print("Directory " , DST_PATH ,  " already exists")  

w = 1920
h = 1080
pond1 = 0.8
pond2 = 0.85
width = pond1*pond2*w
heigth = pond1*pond2*h
stream = False
min_score_thresh = 0.65
faces = {}
face_cnt = 0
init_cnt_max = 40
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0,255,0)
thickness = 2
window = 10
personas_out = None

ts = []

PATH_TO_SAVED_MODEL = join(ROOT_DIR, 'models', 'SSD_v2_fpnlite_640x640_WF5_V2', 'saved_model')
print('Cargando modelo...', end='')
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Listo! Tiempo: {} s'.format(elapsed_time))

def set_stream(value):
    global stream
    stream = value

def set_var():
    global min_score_thresh
    min_score_thresh = float(spin_temp.get()[0:-2])/100

def bb_intersection_over_union(boxA, boxB, overleapThreshold):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iouA = interArea/boxAArea
    iouB = interArea/boxBArea
    if iouA > overleapThreshold:
        return iouA
    elif iouB > overleapThreshold: 
        return iouB
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def adjust_box(boxA, boxB):
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    return xA, yA, xB, yB

def detect(img):
    global min_score_thresh, GPU_memory

    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    if GPU:
        GPU_memory.append(tf.config.experimental.get_memory_info('GPU:0')['peak']*10. ** -9)
        
    scores = detections['detection_scores'][0]
    boxes = detections['detection_boxes'][0]
    new_boxes = []
    new_scores = []
    for i in range(boxes.shape[0]):
        if scores is None or scores[i] > min_score_thresh:
            y1 = int(boxes[i][0]*heigth)
            x1 = int(boxes[i][1]*width)
            y2 = int(boxes[i][2]*heigth)
            x2 = int(boxes[i][3]*width)
            sc = np.round(float(scores[i]),2)
            new_boxes.append([x1,y1,x2,y2])
            new_scores.append(sc)
    return new_boxes, new_scores

def select_boxes(boxes, scores):
    global faces, face_cnt
    for f in faces:
        faces[f]['cnt'] += 1
        
    for i, bA in enumerate(boxes):
        scA = scores[i]
        if not faces:
            x1,y1,x2,y2 = bA
            nx1 = (x1-window) if (x1-window) >= 0 else 0
            ny1 = (y1-window) if (y1-window) >= 0 else 0
            nx2 = (x2+window) if (x2+window) <= w else w
            ny2 = (y2+window) if (y2+window) <= h else h
            bA = [nx1,ny1,nx2,ny2]
            faces['Face_' + str(face_cnt)] = {'Box': bA, 'Score': scA, 'cnt': 0, 'out': None, 'init_cnt': 0}
            face_cnt += 1
        else:
            flag = 0
            for f in faces:
                data = faces[f]
                bB = data['Box']
                scB = data['Score']
                init_cnt = data['init_cnt']
                iou = bb_intersection_over_union(bA,bB,0.2)
                if iou > 0.2:
                    flag = True
                    faces[f]['Score'] = np.round((scA+scB)/2,2)
                    faces[f]['cnt'] = 0
                    faces[f]['init_cnt'] = init_cnt + 1
                    if init_cnt < init_cnt_max:
                        box = adjust_box(bA, bB)
                        faces[f]['Box'] = box
                    break

            if not flag:
                x1,y1,x2,y2 = bA
                nx1 = (x1-window) if (x1-window) >= 0 else 0
                ny1 = (y1-window) if (y1-window) >= 0 else 0
                nx2 = (x2+window) if (x2+window) <= w else w
                ny2 = (y2+window) if (y2+window) <= h else h
                bA = [nx1,ny1,nx2,ny2]
                faces['Face_' + str(face_cnt)] = {'Box': bA, 'Score': scA, 'cnt': 0, 'out': None, 'init_cnt': 0}
                face_cnt += 1

    return True

# function for video streaming
def video_stream():
    global stream, faces, w, h, personas_out, ts, CPU_usage, RAM_usage, GPU_memory
    if stream:
        cv2image = np.array(ImageGrab.grab(bbox=(0,0,w,h)))
        img = Image.fromarray(cv2image)
        img = img.resize((int(pond1*pond2*w), int(pond1*pond2*h)), Image.ANTIALIAS)
        img = np.array(img)
        t1 = time.time()
        boxes, scores = detect(img)
        select_boxes(boxes, scores)
        t2 = time.time()
        print('Tiempo detección y procesado: ', t2-t1)
        ts.append(t2 - t1)
        
        img_disp = img
        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        to_pop = []
        
        for f in faces:
            data = faces[f]
            x1,y1,x2,y2 = data['Box']
            sc = data['Score']
            cnt = data['cnt']
            init_cnt = data['init_cnt']
            if cnt <= 40:
                if (init_cnt >= init_cnt_max):
                    if (data['out'] == None):
                        [h_im, w_im] = img_BGR[y1:y2,x1:x2].shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(join(DST_PATH, f + '.avi'), fourcc, 10, (w_im, h_im))
                        faces[f]['out'] = out
                    else:
                        faces[f]['out'].write(img_BGR[y1:y2,x1:x2])

            
                img_disp = cv2.putText(img_disp, str(int(sc*100)) + '%', (x1,y1), font, fontScale, color, thickness, cv2.LINE_AA)
                cv2.rectangle(img_disp, (x1,y1), (x2,y2), color, 2)
            else:
                to_pop.append(f)

        for f in to_pop:
            faces.pop(f)

        if personas_out == None:
            [h_im, w_im] = img_disp.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            personas_out = cv2.VideoWriter(join(DST_PATH, 'personas.avi'), fourcc, 10, (w_im, h_im))
        else:
            personas_out.write(cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR))


        img_disp = Image.fromarray(img_disp)
        imgtk = ImageTk.PhotoImage(image=img_disp)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        CPU_usage.append(python_process.cpu_percent()/6.)
        RAM_usage.append(python_process.memory_info()[0]/2.**30)


    lmain.after(10, video_stream)




root = Tk()
root.geometry(str(int(w*pond1)) + 'x' + str(int(h*pond1)))
root.configure(bg = 'white')
root.title('Aplicación detector de rostros')
entry_var = StringVar()

# Pack
fm = Frame(root)
ttk.Button(fm, text='Iniciar', command=lambda *args: set_stream(True)).pack(side='left', padx=5, pady=5)
ttk.Button(fm, text='Detener', command=lambda *args: set_stream(False)).pack(side='left', padx=5, pady=5)
spin_temp = ttk.Spinbox(fm, from_=0, to=100, increment=1, format="%0.1f%%")
spin_temp.insert(0, str(min_score_thresh*100) + "%")
spin_temp.pack(side='left', padx=5, pady=5)
ttk.Button(fm, text='Set', command=lambda *args: set_var()).pack(side='left', padx=5, pady=5)
ttk.Button(fm, text='Salir', command=root.destroy).pack(side='left', padx=5, pady=5)

fm.pack(side='top')

# Create a frame
app = Frame(root, bg="white")
app.pack()

# Create a label in the frame
lmain = Label(app)
lmain.pack()

video_stream()
root.mainloop()


print('Mean time per frame: ', np.mean(ts))
print('CPU mean usage %: ', np.mean(CPU_usage))
print('RAM mean usage GB: ', np.mean(RAM_usage))
if GPU:
    print('GPU mean usage : ', np.mean(GPU_memory))