
from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation, Dropout, Input, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.utils import np_utils

import matplotlib.pyplot as plt
import sys, os

import numpy as np
from PIL import Image
from sklearn.cross_validation import train_test_split

img_rows = 64
img_cols = 64
channels = 3

optimizer_G = Adam(0.0002, 0.5)
optimizer_D = Adam(0.0002, 0.5)

class GAN():
    def __init__(self):

        # Initialize 
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Build the discriminator
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(inputs=z, outputs=validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_G)


    def build_generator(self):
        drp = 0.25
        input_shape = (self.img_rows, self.img_cols, self.channels)
        mom=0.8
        
        generator = Sequential()
        generator.add(Dense(units= 512*4*4, kernel_initializer='glorot_uniform', input_dim=100))
        generator.add(Reshape(target_shape=(4, 4, 512)))
        generator.add(BatchNormalization(momentum=mom))
#         generator.add(Activation('relu'))
        generator.add(LeakyReLU(0.2))
        generator.add(Dropout(drp))
    
#         generator.add(Conv2DTranspose(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same',
#                                       data_format='channels_last',
#                                       kernel_initializer='glorot_uniform'))
#         generator.add(BatchNormalization(momentum=0.5))
#         generator.add(Activation('relu'))
        
        generator.add(Conv2DTranspose(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_uniform'))
        generator.add(BatchNormalization(momentum=mom))
#         generator.add(Activation('relu'))
        generator.add(LeakyReLU(0.2))
        generator.add(Dropout(drp))

        generator.add(Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_uniform'))
        generator.add(BatchNormalization(momentum=mom))
#         generator.add(Activation('relu'))
        generator.add(LeakyReLU(0.2))
        generator.add(Dropout(drp))

        generator.add(Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_uniform'))
        generator.add(BatchNormalization(momentum=mom))
#         generator.add(Activation('relu'))
        generator.add(LeakyReLU(0.2))
        generator.add(Dropout(drp))

        generator.add(Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                      data_format='channels_last',
                                      kernel_initializer='glorot_uniform'))
        generator.add(Activation('tanh'))
        print("Generator: ")
        generator.summary()

        # optimizer = Adam(lr=0.00015, beta_1=0.5)
        generator.compile(loss='binary_crossentropy', optimizer=optimizer_D, metrics=None)

        return generator

    def build_discriminator(self):
        drp = 0.25
        mom=0.8
        
        discriminator = Sequential()
        discriminator.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform',
                                 input_shape=self.img_shape))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(drp))

        discriminator.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform'))
        discriminator.add(BatchNormalization(momentum=mom))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(drp))

        discriminator.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform'))
        discriminator.add(BatchNormalization(momentum=mom))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(drp))

        discriminator.add(Conv2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                 data_format='channels_last',
                                 kernel_initializer='glorot_uniform'))
        discriminator.add(BatchNormalization(momentum=mom))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(drp))

        discriminator.add(Flatten())
        discriminator.add(Dense(1))
        discriminator.add(Activation('sigmoid'))
        print("Discriminator: ")
        discriminator.summary()

        # optimizer = Adam(lr=0.0002, beta_1=0.5)
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_D, metrics=['accuracy'])

        return discriminator

    def load_data(self):
#         path = 'pizza/pizza'
        path = 'faces1'

        num_samples=len(os.listdir(path))
        print(num_samples)
        imlist = os.listdir(path)

        immatrix = np.array([np.array(Image.open(path + '/' + im2).resize((img_rows, img_cols))).flatten() for im2 in imlist], 'f')
        label=np.ones((num_samples,),dtype = int)
        label[0:] = 0      
        train_data = [immatrix,label]
        nb_classes = 2 

        X_train, X_test, y_train, y_test = train_test_split(train_data[0], train_data[1], test_size=0.1, random_state=4)
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        return X_train, X_test, Y_train, Y_test

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train, _, _, _ = self.load_data()

        # Rescale -1 to 1
        X_train /= 128.0
        X_train -= 1.0
#         X_train = X_train / 127.5 - 1.

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        d_loss_all, g_loss_all = [], []

        for epoch in range(epochs):

            #  Train Discriminator

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            # print(idx)

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
#             print(imgs.shape, gen_imgs.shape, valid.shape, fake.shape)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #  Train Generator

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
  
            d_loss_all.append(d_loss[0])
            g_loss_all.append(g_loss)
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            
        fig, ax = plt.subplots()
#             losses = np.array(losses)
        plt.plot(d_loss_all, label='Discriminator', alpha=0.5)
        plt.plot(g_loss_all, label='Generator', alpha=0.5)
        plt.title("Training Losses")
        plt.legend()
        if epoch == 0:
          plt.legend()
        plt.pause(0.0000000001)
        plt.show()
        plt.savefig('trainingLossPlot.png')
        
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs += 1.0
        gen_imgs *= 128.0
#         gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.show()
        plt.close()

r, c = 5, 5
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(X_train[cnt, :,:,:])
        axs[i,j].axis('off')
        cnt += 1
# fig.savefig("images/%d.png" % epoch)
plt.show()
plt.close()

# start 
gan = GAN()
gan.train(epochs=1000, batch_size=128, sample_interval=100)
















