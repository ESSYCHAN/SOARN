import numpy as np
import pickle
from PIL import Image


new_h, new_w = 256, 256


def prep_imgs_data(dpath: str = None, IMG_SIZE: int = new_h):
    img_array = Image.open(dpath)
    img_array = img_array.convert('RGB')
    new_array = img_array.resize((IMG_SIZE, IMG_SIZE))
#     print(new_array)
    training_data = np.array(new_array)
    # print(training_data)

    X = np.moveaxis(np.array(training_data).reshape(
        IMG_SIZE, IMG_SIZE, 3), -1, 0)
    pickle_out = open('X.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    Train_data = pickle.load(open('X.pickle', 'rb'))

    return Train_data


def load_data(data_path: str = None):
    #     data_path = [r"/Users/esthermulwa/soarn/Data/Test/triangle/threesome-g6d95f4e4e_1280.png"]

    x_train = []
    for file_names in data_path:
        x_train.append(prep_imgs_data(dpath=file_names, IMG_SIZE=new_h))
    X_train_batch = np.stack(x_train)
    return X_train_batch