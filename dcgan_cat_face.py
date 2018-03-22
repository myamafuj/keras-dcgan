from keras.engine.topology import Input
from keras.layers.core import Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model


def build_generator():
    noise = Input(shape=(100,))

    x = Reshape((1, 1, 100))(noise)

    x = Conv2DTranspose(512, 2, strides=1, padding='valid')(x)
    x = BatchNormalization(momentum=.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=.9)(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(3, 4, strides=2, padding='same')(x)
    fake_image = Activation('tanh')(x)

    model = Model(noise, fake_image)
    return model


def build_discriminator():
    image = Input(shape=(32, 32, 3))

    x = Conv2D(64, 4, strides=2, padding='same')(image)
    x = BatchNormalization(momentum=.9)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=.9)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=.9)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(512, 4, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=.9)(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1, 2, strides=1, padding='valid')(x)
    x = GlobalAveragePooling2D()(x)
    p = Activation('sigmoid')(x)

    model = Model(image, p)
    return model


def build_gan(generator, discriminator):
    # compile 前に変更しておくだけで OK
    discriminator.trainable = False

    noise = Input(shape=(100,))

    fake_image = generator(noise)
    p = discriminator(fake_image)

    model = Model(noise, p)
    return model


def build_train_discriminator(discriminator):
    # compile 前に変更しておくだけで OK
    discriminator.trainable = True

    real_image = Input(shape=(32, 32, 3))
    fake_image = Input(shape=(32, 32, 3))

    p_r = discriminator(real_image)
    p_f = discriminator(fake_image)

    model = Model([real_image, fake_image], [p_r, p_f])
    return model


def main():
    import os
    import time
    import json

    import matplotlib.pyplot as plt
    import numpy as np
    from keras.losses import binary_crossentropy
    from keras.optimizers import Adam
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    train_x = np.load('./data/cat_face.npy')

    # -1 から 1 に変換
    train_x = (train_x.astype('float32')-127.5)/127.5

    train_size = train_x.shape[0]

    generator = build_generator()
    print('Generator model:')
    print(generator.summary())

    discriminator = build_discriminator()
    print('Discriminator model:')
    print(discriminator.summary())

    gan = build_gan(generator, discriminator)
    gan.compile(
        loss=binary_crossentropy,
        optimizer=Adam(lr=2e-4, beta_1=.5),
        metrics=['accuracy']
    )

    train_discriminator = build_train_discriminator(discriminator)
    train_discriminator.compile(
        loss=binary_crossentropy,
        optimizer=Adam(lr=2e-4, beta_1=.5),
        metrics=['accuracy']
    )

    batch_size = 128
    epochs = 200
    steps_per_epoch = train_size//batch_size
    # steps_per_epoch = 500

    print(
        f'Train size: {train_size}, '
        f'Batch size: {batch_size}, '
        f'Epochs: {epochs}, '
        f'Total Steps: {steps_per_epoch*epochs}'
    )

    p_r = np.ones((batch_size, 1), dtype=np.float32)
    p_f = np.zeros((batch_size, 1), dtype=np.float32)

    np.random.seed(0)
    test_noise = np.random.uniform(-1, 1, size=(30, 100)).astype(np.float32)
    test_indices = np.random.randint(0, train_size, size=6)
    plot_real_images = train_x[test_indices]
    np.random.seed(None)

    d_loss, d_acc = [], []
    g_loss, g_acc = [], []

    cnt = 0
    for epoch in range(epochs):

        print(f'Epoch {epoch+1}/{epochs}')
        start = time.time()

        # データをシャッフル
        np.random.shuffle(train_x)

        d_loss_epoch, d_acc_epoch = [], []
        g_loss_epoch, g_acc_epoch = [], []

        for step in range(steps_per_epoch):

            # バッチサイズの分だけ画像を選択
            real_images = train_x[step*batch_size:(step+1)*batch_size]

            # バッチサイズの分だけランダムにノイズを生成
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            noise = noise.astype(np.float32)

            # generatorにより画像を生成
            fake_images = generator.predict(noise)

            # Discriminatorのtrain
            d_history = train_discriminator.train_on_batch(
                [real_images, fake_images], [p_r, p_f]
            )
            tmp_d_loss = float(d_history[0]/2)
            tmp_d_acc = float((d_history[3]+d_history[4])/2)
            d_loss.append(tmp_d_loss)
            d_acc.append(tmp_d_acc)
            d_loss_epoch.append(tmp_d_loss)
            d_acc_epoch.append(tmp_d_acc)

            # バッチサイズの分だけランダムにノイズを生成
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            noise = noise.astype(np.float32)

            # Generatorのtrain
            g_history = gan.train_on_batch(noise, p_r)
            tmp_d_loss = float(g_history[0])
            tmp_d_acc = float(g_history[1])
            g_loss.append(tmp_d_loss)
            g_acc.append(tmp_d_acc)
            g_loss_epoch.append(tmp_d_loss)
            g_acc_epoch.append(tmp_d_acc)

            cnt += 1

        d_loss_std = np.std(d_loss_epoch)
        g_loss_std = np.std(g_loss_epoch)

        d_loss_mean = np.mean(d_loss_epoch)
        g_loss_mean = np.mean(g_loss_epoch)

        d_acc_std = np.std(d_acc_epoch)
        g_acc_std = np.std(g_acc_epoch)

        d_acc_mean = np.mean(d_acc_epoch)
        g_acc_mean = np.mean(g_acc_epoch)

        print(
            f'd_loss: {d_loss_mean:.4f}, '
            f'd_loss_std: {d_loss_std:.4f}, '
            f'd_acc: {d_acc_mean:.2f}, '
            f'd_acc_std: {d_acc_std:.2f}, '
        )

        print(
            f'g_loss: {g_loss_mean:.4f}, '
            f'g_loss_std: {g_loss_std:.4f}, '
            f'g_acc: {g_acc_mean:.2f}, '
            f'g_acc_std: {g_acc_std:.2f}, '
        )

        print(
            f'time: {int(time.time() - start)} s'
        )

        generated_images = generator.predict(test_noise)

        plt.figure(figsize=(6, 6))
        grid0 = GridSpec(1, 2, width_ratios=(5, 1))
        grid00 = GridSpecFromSubplotSpec(6, 5, subplot_spec=grid0[0])
        for i in range(generated_images.shape[0]):
            plt.subplot(grid00[i])
            img = generated_images[i, :, :, :]*127.5 + 127.5
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

        grid01 = GridSpecFromSubplotSpec(6, 1, subplot_spec=grid0[1])
        for i in range(plot_real_images.shape[0]):
            plt.subplot(grid01[i])
            img = plot_real_images[i, :, :, :]*127.5 + 127.5
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                            hspace=0, wspace=0)

        path = f'figure/dcgan_cat_face/image_{epoch+1:03d}.png'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        plt.savefig(path)
        plt.close()

        path = f'var/log/dcgan_cat_face_history.json'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        history = [d_loss, g_loss]
        with open(path, 'w+', encoding='UTF-8') as f:
            json.dump(history, f)

        path = f'model/dcgan_cat_face/model_{epoch+1:03d}.h5'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        generator.save(path)


if __name__ == '__main__':
    main()
