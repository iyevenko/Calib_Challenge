import numpy
import cv2
import os


DATA_PATH = 'calib_challenge/labeled/'


def extract_jpgs():
    mov_path = DATA_PATH
    jpg_path = 'jpgs/'

    for mov in os.listdir(mov_path):
        mov_name, ext = os.path.splitext(mov)
        if ext != '.hevc':
            continue

        frame_dir = jpg_path + mov_name + '/'
        if not os.path.exists(frame_dir):
            os.mkdir(frame_dir)

        count = 0
        cap = cv2.VideoCapture(mov_path + mov)
        success, frame = cap.read()
        while success:
            frame_path = frame_dir + f'frame_{count:04}.jpg'
            cv2.imwrite(frame_path, frame)

            success, frame = cap.read()
            count += 1

        cap.release()


if __name__ == '__main__':
    extract_jpgs()