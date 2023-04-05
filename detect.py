from backend import DefectDetector
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter
from tkinter import filedialog as file_dlg
from tkinter.messagebox import showwarning
import tkinter.font as font
from PIL import Image as PILImage
from PIL import ImageTk as PILImageTk
import multiprocessing as mp


IMAGE_SIZE = (600, 600)
win_main = tkinter.Tk()
img_queue = mp.Queue(10)
result_queue = mp.Queue(10)
stop = mp.Event()


def detect_thread(in_queue, out_queue, stop_event):
    defect_detector = DefectDetector.DefectDetector('./mask_rcnn_model.resnet101.h5')
    while not stop_event.is_set():
        try:
            img = in_queue.get(block=True, timeout=0.1)
            image = cv2.resize(img, IMAGE_SIZE)
            img_re, result = defect_detector.detect(image)
            out_queue.put((img_re, result,))
        except Exception as e:
            pass


def image_t0_tk(img):
    blue, green, red = cv2.split(cv2.resize(img, (400, 400)))
    img = cv2.merge((red, green, blue))
    image_pil = PILImage.fromarray(img)
    return PILImageTk.PhotoImage(image=image_pil)


detect_thr = mp.Process(target=detect_thread, args=(img_queue, result_queue, stop,))
label_image_orig = image_t0_tk(np.zeros((400, 400, 3), dtype=np.uint8))
label_image_processed = image_t0_tk(np.zeros((400, 400, 3), dtype=np.uint8))


def on_closing():
    global stop
    stop.set()
    detect_thr.join()
    win_main.destroy()


def load_image():
    global label_image_orig
    global label_image_processed
    file_types = (
        ('Images', '*.jpg *jpeg *.png *.tiff'),
        ('All files', '*.*'),
    )
    file_name = file_dlg.askopenfilename(
        title='Choose image',
        filetypes=file_types
    )
    if len(file_name) > 0:
        try:
            image = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception as e:
            showwarning(
                title='Error!',
                message='Unsupported image file!'
            )
            return
        img_queue.put(image)
        img_ret, result = result_queue.get(block=True)
        label_image_orig = image_t0_tk(image)
        label_image_processed = image_t0_tk(img_ret)
        img_label_orig.config(image=label_image_orig)
        img_label_processed.config(image=label_image_processed)
        txt_label.config(text='Total {} defects found'.format(len(result['coords'])))
        win_main.update_idletasks()


if __name__ == '__main__':
    detect_thr.start()
    win_main.geometry('820x490')
    win_main.title('Defect detector')
    start_btn = tkinter.Button(text='Detect', font='Helvetica 16 bold')
    start_btn.config(command=load_image)
    start_btn.place(x=20, y=430, width=100, height=30)
    img_label_orig = tkinter.Label(win_main, borderwidth=5, relief="raised")
    img_label_orig.config(image=label_image_orig)
    img_label_orig.place(x=10, y=10, width=400, height=400)
    img_label_processed = tkinter.Label(win_main, borderwidth=5, relief="raised")
    img_label_processed.config(image=label_image_processed)
    img_label_processed.place(x=420, y=10, width=400, height=400)
    txt_label = tkinter.Label(win_main, text='', borderwidth=2, relief="sunken", font='Helvetica 18 bold')
    txt_label.place(x=150, y=430, width=300, height=30)
    win_main.protocol('WM_DELETE_WINDOW', on_closing)
    win_main.mainloop()
