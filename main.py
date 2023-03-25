import torch
import numpy as np
import cv2
import pafy
import time
import redis
import math
import csv
from datetime import datetime

x_real = 160 # cm
y_real = 80 # cm
x_pix = 1280
y_pix = 720
cmpp_x = x_real / x_pix
cmpp_y = y_real / y_pix


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using OpenCV.
    """
    
    def __init__(self, img_path=None, video_path=None, video_out=None, realtime=None):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.img_path = img_path
        self.video_path = video_path
        self.video_out = video_out
        self.realtime = realtime
        self.model = self.load_model()
        self.classes = self.model.names
        self.track = []
        self.time_stamp = []
        self.speedCMPS = None
        self.cy1 = 200
        self.offset = 6
        self.counter = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)

    def calculate_speed(self, estimated_speed):
        self.speedCMPS = round(sum(estimated_speed) / len(estimated_speed), 3)

    
    def points(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            colorBGR = [x, y]
            print(colorBGR)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        print(f"[INFO] Loading model... ")
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s_arowana_v7.pt')
        model = torch.hub.load('/Users/sahachai/Documents/yolo_project/yolov5', 
                               'custom',
                                path='yolov5s.pt',
                                # path='yolov5s-cbam-bb.pt',
                                source='local')
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.conf = 0.5  # NMS confidence threshold
        # model.iou = 0.45
        return model


    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        # print(f"[INFO] Detecting. . . ")
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = self.model(frame, size=1280)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        return results


    def plot_boxes(self, results, frame, area=0):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        # self.track = []
        cx = 0
        cy = 0
        for index, row in results.pandas().xyxy[0].iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            d = (row['name'])
            cx = int(x1+x2) // 2
            cy = int(y1+y2) // 2

            # if len(self.track) >= 15:
            #     self.track.pop(0)
                # self.time_stamp.pop(0)
            
            # re = cv2.pointPolygonTest(np.array(area, np.int32), ((cx,cy)), False)
            # if re >= 0:
            # self.track.append((cx, cy))
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, str(d), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2)
            # cv2.circle(frame, (cx,cy), 5, (255,0,0), -1)
            # for t in self.track:
            #     cv2.circle(frame, t, radius=2, color=(0,0,255), thickness=-1)
            # self.track.append([cx])
            # if cy < (self.cy1 - self.offset):
            #     cv2.line(frame, (1,200), (630,200), (0,0,255), 2)
                # self.counter += 1
                # print(self.counter)
        return frame, [cx,cy]


    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        
        if self.img_path != None:
            print(f"[INFO] Working with image: {self.img_path}")
            frame = cv2.imread(self.img_path)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            results = self.score_frame(frame)

            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            frame = self.plot_boxes(results, frame)

            cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)

            while True:
                cv2.imshow("img_only", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"[INFO] Exiting. . . ")
                    # cv2.imwrite("cat_output_02.jpg",frame) ## if you want to save he output result.
                    break
        
        elif self.video_path != None:
            print(f"[INFO] Working with video: {self.video_path}")

            cv2.namedWindow('video_out')
            cv2.setMouseCallback('video_out', self.points)
            start_time = time.time()
            display_time = 2
            fc = 0
            FPS = 0
            cap = cv2.VideoCapture(self.video_path)
            area = [(1,181), (1,465), (239,469), (480,289), (488,169)]
            count = 0
            
            if self.video_out: ### creating the video writer if video output path is given

                # by default VideoCapture returns float instead of int
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
                out = cv2.VideoWriter(self.video_out, codec, fps, (width, height))
            with open('result.csv', 'w') as file:
                # writer = csv.writer(file)
                # writer.writerow(['index', 'x', 'y', 'timestamp'])
                i = 0
                while cap.isOpened():   
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # count += 1
                    # if count % 4 != 0:
                    #     continue
                    frame = cv2.resize(frame, (640, 480))
                    results = self.score_frame(frame)                  
                    frame, coord = self.plot_boxes(results, frame, area)
                    # print(coord)
                    # writer.writerow([i, coord[0], coord[1], datetime.now()])
                    fc += 1
                    TIME = time.time() - start_time
                    if (TIME) >= display_time:
                        FPS = fc / (TIME)
                        # print(FPS)
                        fc = 0
                        start_time = time.time()
                    fps_disp = "FPS: " + str(FPS)[:5]
                    cv2.putText(frame, fps_disp, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    cv2.polylines(frame, [np.array(area, np.int32)], True, (0,0,255), 2)
                    # cv2.line(frame, (1,200), (639,200), (0,0,255), 2)
                    # print(len(self.track))
                    # count_disp = 'Count: ' + str(len(self.track))
                    # for t in self.track:
                    #     cv2.circle(frame, t, radius=2, color=(0,0,255), thickness=-1)
                    # cv2.putText(frame, count_disp, (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    cv2.imshow("video_out", frame)
                    i += 1

                    if self.video_out:
                        print(f"[INFO] Saving output video. . . ")
                        out.write(frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
                if self.video_out:
                    out.release()
                cv2.destroyAllWindows()

        elif self.realtime != None:
            # cv2.namedWindow('video_out')
            # cv2.setMouseCallback('video_out', self.points)
            r = redis.Redis(host='localhost', port=6379, db=0)
            start_time = time.time()
            display_time = 2
            fc = 0
            FPS = 0
            # area = [(1,181), (1,465), (239,469), (480,289), (488,169)]
            pre_picture = r.get('base64')
            if self.video_out: ### creating the video writer if video output path is given
                # by default VideoCapture returns float instead of int
                width = 640
                height = 480
                codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
                out = cv2.VideoWriter(self.video_out, codec, 5, (width, height))
            # with open('result.csv', 'w') as file:
                # writer = csv.writer(file)
                # writer.writerow(['index', 'x', 'y', 'timestamp'])
                # i = 0
            while True:
                picture = r.get('base64')
                if picture == pre_picture:
                    continue
                pre_picture = picture
                # picdecode = base64.standard_b64decode(picture)
                nparr = np.frombuffer(picture, np.uint8)
                frame = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)
                results = self.score_frame(frame)
                frame, coord = self.plot_boxes(results, frame)
                # writer.writerow([i, coord[0], coord[1], datetime.now()])
                # fc += 1
                # TIME = time.time() - start_time
                # if (TIME) >= display_time:
                #     FPS = fc / (TIME)
                #     # print(FPS)
                #     fc = 0
                #     start_time = time.time()
                # fps_disp = "FPS: " + str(FPS)[:5]
                TIME = time.time() - start_time
                FPS = 1 / (TIME)
                start_time = time.time()
                # fps_disp = "FPS: " + str(FPS)[:5]
                print(FPS)
                # cv2.putText(frame, fps_disp, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                # cv2.polylines(frame, [np.array(area, np.int32)], True, (0,0,255), 2)
                # for t in self.track:
                #     cv2.circle(frame, t, radius=2, color=(0,0,255), thickness=-1)
                cv2.imshow("video_out", frame)
                # i += 1
                # Press Q on keyboard to stop recording
                if self.video_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if self.video_out:
                out.release()
            cv2.destroyAllWindows()



# Create a new object and execute.
# detection = ObjectDetection(img_path="test_cat_02.jpeg")
# detection = ObjectDetection(video_path="/Users/sahachai/Documents/arowana_hatyai/arowana/VGA14.mp4", video_out="runs/trej.mp4")
# detection = ObjectDetection(video_path="test_pos.mp4")
detection = ObjectDetection(realtime=True)
# detection = ObjectDetection(video_path="http://172.20.10.4:80/image")
# detection = ObjectDetection(video_path=0)

detection()