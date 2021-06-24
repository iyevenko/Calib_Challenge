import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

DATA_PATH = 'calib_challenge/labeled/'

def plot_and_play():
    for file_name in os.listdir(DATA_PATH):
        if file_name[-3:] == 'txt':
            file_path = DATA_PATH + file_name
            with open(file_path, 'r') as f:
                data = f.readlines()
                data = [[float(x) for x in line.split(' ')] for line in data]
                plt.title(file_name)
                plt.plot(data)
                plt.show()
        else:
            file_path = DATA_PATH + file_name
            cap = cv2.VideoCapture(file_path)

            if not cap.isOpened():
                print(f'Error reading {file_name}')

            ret = True
            while cap.isOpened() and ret:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow(file_name, frame)
                    if cv2.waitKey(1000//30) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    idx = 3
    label_fname = f'{idx}.txt'
    video_fname = f'{idx}.hevc'
    label_path = DATA_PATH + label_fname
    video_path = DATA_PATH + video_fname

    with open(label_path, 'r') as f:
        data = f.readlines()
        data = [[float(x) for x in line.split(' ')] for line in data]
        pred_data = np.load(DATA_PATH + '3.npy')
        # plt.title(label_fname)
        # plt.plot(data)
        # plt.show()

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f'Error reading {video_path}')

        ret = True
        i = 0
        R = 910
        scale = 10000
        while cap.isOpened() and ret:
            ret, frame = cap.read()
            if ret:
                H, W, C = frame.shape
                pitch, yaw = np.nan_to_num(data[i])
                x = int(W/2 + R*np.sin(yaw))
                y = int(H/2 + R*np.sin(pitch))
                pred_x = int(W/2 + scale*pred_data[i,0])
                pred_y = int(H/2 + scale*pred_data[i,1])
                cv2.drawMarker(frame, (W//2, H//2), (255,0,0), cv2.MARKER_CROSS, 10)
                cv2.drawMarker(frame, (x, y), (0,255,0), cv2.MARKER_CROSS, 10)
                cv2.drawMarker(frame, (pred_x, pred_y), (0,0,255), cv2.MARKER_CROSS, 10)
                cv2.imshow(video_fname, frame)
                if cv2.waitKey(1000//30) & 0xFF == ord('q'):
                    break
            i+=1

        cap.release()
        cv2.destroyAllWindows()
