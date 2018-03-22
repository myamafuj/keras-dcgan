import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


def main():
    epoch = 300
    path = f'model/dcgan_cat_face_mod/model_{epoch:03d}.h5'
    generator = load_model(path)
    np.random.seed(1)
    noise = np.random.randn(100).astype(np.float32)
    test_noise = np.zeros((36, 100), dtype=np.float32)
    for i in range(36):
        test_noise[i] = noise
    for i, val in enumerate(np.arange(-3, 3, 6/36)):
        test_noise[i, 49] = val
    generated_images = generator.predict(test_noise)
    plt.figure(figsize=(6, 6))
    for i in range(36):
        plt.subplot(6, 6, i+1)
        img = generated_images[i, :, :, :]*127.5 + 127.5
        img = img.astype(np.uint8)
        plt.imshow(img)
        plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                        hspace=0, wspace=0)
    path = f'figure/dcgan_cat_face_mod_z.png'
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    main()
