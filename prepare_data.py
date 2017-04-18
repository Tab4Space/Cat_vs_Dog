"""
1. Read file list
2. Read image using cv2
3. Resize image
4. Restore image(64x64 size and gray scale)
"""

import csv, cv2, os
from tqdm import tqdm


ORIGIN_TRAIN_DATA_PATH = 'input your data path'
ORIGIN_TEST_DATA_PATH = 'input your data path'

RESIZE_TRAIN_DATA_PATH = 'input your data path'
RESIZE_TEST_DATA_PATH = 'input your data path'

TRAIN_LABEL_CSV_PATH = 'input your data path'


### Prepare image data for training
def prepre_resize_train():
    ### Create directory
    if os.path.exists(RESIZE_TRAIN_DATA_PATH):
        print "Already exist resize train directory"
    else:
        os.mkdir(RESIZE_TRAIN_DATA_PATH)

        for data in tqdm(os.listdir(ORIGIN_TRAIN_DATA_PATH)):
            img = cv2.resize(cv2.imread(os.path.join(ORIGIN_TRAIN_DATA_PATH, data), cv2.IMREAD_GRAYSCALE), (64, 64))
        
            if len(data.split('.')[1]) == 1:
                file_name = data.replace(data.split('.')[1], '0000'+data.split('.')[1])
                cv2.imwrite(RESIZE_TRAIN_DATA_PATH+'/'+file_name, img)

            elif len(data.split('.')[1]) == 2:
                file_name = data.replace(data.split('.')[1], '000'+data.split('.')[1])
                cv2.imwrite(RESIZE_TRAIN_DATA_PATH+'/'+file_name, img)

            elif len(data.split('.')[1]) == 3:
                file_name = data.replace(data.split('.')[1], '00'+data.split('.')[1])
                cv2.imwrite(RESIZE_TRAIN_DATA_PATH+'/'+file_name, img)

            elif len(data.split('.')[1]) == 4:
                file_name = data.replace(data.split('.')[1], '0'+data.split('.')[1])
                cv2.imwrite(RESIZE_TRAIN_DATA_PATH+'/'+file_name, img)


def prepare_resize_test():
    ### Create directory
    if os.path.exists(RESIZE_TEST_DATA_PATH):
        print "Already exist resize test directory"
    else:
        os.mkdir(RESIZE_TEST_DATA_PATH)

        for data in tqdm(os.listdir(ORIGIN_TEST_DATA_PATH)):
            img = cv2.resize(cv2.imread(os.path.join(ORIGIN_TEST_DATA_PATH, data), cv2.IMREAD_GRAYSCALE), (64, 64))
        
            if len(data.split('.')[0]) == 1:
                file_name = data.replace(data.split('.')[0], '0000'+data.split('.')[0])
                cv2.imwrite(RESIZE_TEST_DATA_PATH+'/'+file_name, img)
            
            elif len(data.split('.')[0]) == 2:
                file_name = data.replace(data.split('.')[0], '000'+data.split('.')[0])
                cv2.imwrite(RESIZE_TEST_DATA_PATH+'/'+file_name, img)

            elif len(data.split('.')[0]) == 3:
                file_name = data.replace(data.split('.')[0], '00'+data.split('.')[0])
                cv2.imwrite(RESIZE_TEST_DATA_PATH+'/'+file_name, img)

            elif len(data.split('.')[0]) == 4:
                file_name = data.replace(data.split('.')[0], '0'+data.split('.')[0])
                cv2.imwrite(RESIZE_TEST_DATA_PATH+'/'+file_name, img)


def prepare_csv():
    if os.path.exists(TRAIN_LABEL_CSV_PATH):
        print 'Already csv file exist'
    else:
        image_list = [RESIZE_TRAIN_DATA_PATH+ "/" + file_name for file_name in os.listdir(RESIZE_TRAIN_DATA_PATH)]
        image_list.sort()
        label_list = [0 if label.split('.')[0].split('/')[-1] == 'cat' else 1 for label in image_list]

        with open(TRAIN_LABEL_CSV_PATH, 'wb') as f:
            writer = csv.writer(f, delimiter=',')
            for i in range(len(image_list)):
                writer.writerow([image_list[i], label_list[i]])