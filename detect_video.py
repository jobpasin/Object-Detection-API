import time

import json
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from centroidtracker import CentroidTracker

from performance import TimeMeasure
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video/paris.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('logs', './detections/report2.json', 'path to result logs')


def img_read_wrapper(vid, out_queue: Queue, out_queue2: Queue):
    print("img_read_wrapper: {}".format(threading.current_thread()))
    global stop_threads
    count = 0
    frame_count = 0
    while True:
        _, img = vid.read()

        if img is None or stop_threads:
            logging.warning("Empty Frame:" + str(frame_count))
            time.sleep(0.1)
            count += 1
            if count < 3:
                continue
            else:
                print("Stopeed")
                out_queue.put(None)
                out_queue2.put(None)
                break
        else:
            frame_count += 1
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, FLAGS.size)
            out_queue.put(img_in)
            out_queue2.put(img)


def predict_wrapper(yolo, in_queue: Queue, out_queue: Queue):
    print("prediction_wrapper: {}".format(threading.current_thread()))
    global stop_threads
    fps = 0.0
    while True:
        img_in = in_queue.get()
        if img_in is None or stop_threads:
            out_queue.put(None)
            break

        t1 = time.time()
        with TimeMeasure('Prediction'):
            boxes, scores, classes, nums = yolo.predict(img_in)
        fps = (fps + (1. / (time.time() - t1))) / 2
        output = {'boxes': boxes, 'scores': scores, 'classes': classes, 'nums': nums, 'fps': fps}
        out_queue.put(output)


def display_wrapper(out, FLAGS, in_queue: Queue, in2_queue: Queue):
    print("display_wrapper: {}".format(threading.current_thread()))
    global stop_threads
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    data_log = {}
    frame_count = 0
    ct = CentroidTracker()
    while True:
        data = in_queue.get()
        img = in2_queue.get()
        if data is None or img is None:
            break
        boxes, scores, classes, nums, fps = data['boxes'], data['scores'], data['classes'], data['nums'], data['fps']

        with TimeMeasure('Display frame:' + str(frame_count)):
            img, rects, log = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            img = cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        objects = ct.update(rects)

        if FLAGS.output:
            out.write(img)
            data_log['frame{}'.format(str(frame_count))] = log
        frame_count += 1

        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            stop_threads = True
            break

    with open(FLAGS.logs, 'w') as f:
        json.dump(data_log, f)
    cv2.destroyAllWindows()


processed_img_queue = Queue()
raw_img_queue = Queue()
yolo_result_queue = Queue()

stop_threads = False

def main(_argv):
    print("Start")
    start_time = time.time()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(img_read_wrapper, vid, processed_img_queue, raw_img_queue)
        executor.submit(predict_wrapper, yolo, processed_img_queue, yolo_result_queue)
        display_wrapper(out, FLAGS, yolo_result_queue, raw_img_queue)

    # read_thread = threading.Thread(target=img_read_wrapper, args=(vid, processed_img_queue, raw_img_queue))
    # predict_thread = threading.Thread(target=predict_wrapper, args=(yolo, processed_img_queue, yolo_result_queue))
    # display_thread = threading.Thread(target=display_wrapper, args=(out, FLAGS, yolo_result_queue, raw_img_queue))
    # threads = [read_thread, predict_thread, display_thread]
    # for t in threads:
    #     t.start()
    # for t in threads:
    #     t.join()
    print("FInish", time.time() - start_time)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
