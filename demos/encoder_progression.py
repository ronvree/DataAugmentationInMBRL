

if __name__ == '__main__':
    import cv2
    import os
    from PIL import Image

    import numpy as np

    log_path = '../logs/log_colab_50iter'
    # log_path = '../logs/log_test_mountaincart'

    T = 100

    I = 42

    for i in range(1, I + 1):

        for t in range(1, T + 1):

            img = Image.open(f'{log_path}/iter_{i}_observations/observations_t_{t}.png')

            img = np.array(img)

            img = cv2.resize(img, (128 * 5, 64 * 5))

            cv2.imshow('Encoder', img)
            cv2.waitKey(20)
