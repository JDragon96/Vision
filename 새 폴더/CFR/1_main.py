from functions import *
from mask_model import *
import tensorflow as tf
import numpy as np
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def main():
    # 이미지, 라벨 가져오기
    file_path = ("./mask", "./non_mask")
    test_file_path = ("./test_mask", "./test_nmask")
    train_image, train_label = img_loading(file_path)
    test_image, test_label = test_img_loading(test_file_path)
    print(np.max(train_image[0]), np.min(train_image[0]))
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(128, 128),
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    #model = cnn_model(train_image.shape[1])

    models = tf.keras.models.load_model('./save_model/128model3')
    models.load_weights("./save_cp/128model3")
    model = models

    sequen = tf.keras.Sequential(
        [
            data_augmentation,
            model
        ]
    )


    sequen.compile(
        loss="categorical_crossentropy",
        optimizer='adam',
        metrics=['accuracy']
    )


    model_name = "128model4"

    checkpoint_filepath = './save_cp/' + model_name
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='max',
        save_best_only=True)

    sequen.fit(train_image,
               train_label,
               validation_split=0.2,
               epochs=100,
               batch_size=20,
               callbacks=[model_checkpoint_callback])

    sequen.summary()
    model.save(f"./save_model/{model_name}")

    for i in range(10):
        a = model.predict(test_image[i][np.newaxis, :, :, :])
        print(np.argmax(a))


def test():
    loaded_model = tf.keras.models.load_model('./save_model/128model4')
    test_file_path = ("./test_mask", "./test_nmask")
    test_image, test_label = test_img_loading(test_file_path)

    print(test_label)
    cv2.imshow("Python_Algorithm", test_image[8])
    cv2.waitKey()
    for i in range(10):
        print(loaded_model.predict(test_image)[i])
        print(np.argmax(loaded_model.predict(test_image)[i]))

if __name__ == "__main__":
    main()
    #test()