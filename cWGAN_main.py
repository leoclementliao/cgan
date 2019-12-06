from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Dense, concatenate, Input

## All variables ##
epoch_num = 100 # total epoch to run
BATCH_SIZE = 1024 # batch size of train set
noise_dim = 3 # dimension of noise vector for generator
condition_dim = 3 # dimension of condition vector for generator
gen_dim = 3 # dimension of generator's output vector
num_first = 10_000 # n first data set
D_cycle = 5 # train disctriminator "D_cycle" times in one epoch
steps_show = 2 # update figure per "steps_show" epoches

## Load data ##
def load(file):
    df = pd.read_csv(file, sep=' ', index_col=0,
                     names=['time', 'temp', 'c1', 'c2', 'c3', 'x1', 'x2', 'x3'])
    input_c = df[['c1', 'c2', 'c3']].values
    input_c[:, 2] = np.abs(input_c[:, 2])
    output_x = df[['x1', 'x2', 'x3']].values
    input_c = tf.cast(input_c, tf.float32)
    output_x = tf.cast(output_x, tf.float32)
    return input_c, output_x

PATH = "./CollisionData/"
inp, out = load(PATH+'He_300_rough_collision.dat')

collision_data = tf.concat([inp, out], axis=1)
train_dataset = tf.data.Dataset.from_tensor_slices(collision_data[:num_first,:]).batch(BATCH_SIZE)

## Generator and Discriminator ##
def Generator():
    inp_condition = Input(shape=[condition_dim, ], name='condition_G')
    inp_noise = Input(shape=[noise_dim, ], name='noise')
    X = concatenate([inp_condition, inp_noise], axis=1)
    X = Dense(32, activation='relu')(X)
    X = Dense(32, activation='relu')(X)
    X = Dense(16, activation='relu')(X)
    last = Dense(gen_dim)(X)
    return tf.keras.Model(inputs=[inp_condition, inp_noise], outputs=last, name='Generator')

def Discriminator():
    inp_condition = Input(shape=[condition_dim, ], name='condition_D')
    inp_target = tf.keras.layers.Input(shape=[gen_dim, ], name='target')
    X = concatenate([inp_condition, inp_target], axis=1)
    X = Dense(32, activation='relu')(X)
    X = Dense(32, activation='relu')(X)
    X = Dense(16, activation='relu')(X)
    last = Dense(1)(X)
    return tf.keras.Model(inputs=[inp_condition, inp_target], outputs=last, name='Discriminator')

generator = Generator()
discriminator = Discriminator()

## Generator loss and Discriminator loss ##
lambda_reg = .5
def discriminator_loss(D_real, D_fake, penalty):
    D_loss = tf.reduce_mean(D_fake - D_real + lambda_reg * penalty)
    return D_loss

def generator_loss(D_fake):
    G_loss = -tf.reduce_mean(D_fake)
    return G_loss

## Optimizers ##
generator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5, beta_2=0.9)

## Gradient penalty to the Discriminator loss ##
def penalty_calculation(X_real, G_fake, condition):
    # Create the gradient penalty operations.
    epsilon = tf.random.uniform(shape=tf.shape(X_real), minval=0., maxval=1.)
    interpolation = epsilon * X_real + (1 - epsilon) * G_fake
    with tf.GradientTape() as pena_tape:
        pena_tape.watch(interpolation)
        penalty = (tf.norm(
            pena_tape.gradient(discriminator([condition, interpolation]), interpolation),
            axis=1) - 1) ** 2.0
    return penalty

## Train Generator and Discriminator independently  ##
@tf.function
def train_G(data_batch):
    noise = tf.random.normal([data_batch.shape[0], noise_dim], mean=0.0, stddev=1.0,
                             dtype=tf.dtypes.float32)
    condition = data_batch[:, :3]
    with tf.GradientTape() as gen_tape:
        G_fake = generator([condition, noise], training=True)
        D_fake = discriminator([condition, G_fake], training=True)
        G_loss = generator_loss(D_fake)
    gradients_of_generator = gen_tape.gradient(G_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    return G_loss

@tf.function
def train_D(data_batch):
    noise = tf.random.normal([data_batch.shape[0], noise_dim], mean=0.0, stddev=1.0,
                             dtype=tf.dtypes.float32)
    condition = data_batch[:, :3]
    target = data_batch[:, 3:]
    with tf.GradientTape() as disc_tape:
        G_fake = generator([condition, noise], training=True)
        D_real = discriminator([condition, target], training=True)
        D_fake = discriminator([condition, G_fake], training=True)
        penalty = penalty_calculation(target, G_fake, condition)
        D_loss = discriminator_loss(D_real, D_fake, penalty)
    gradients_of_discriminator = disc_tape.gradient(D_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return D_loss


def train(dataset, epochs, D_cycle=1, steps_show=10):
    list_lim = [[-25, 25], [-25, 25], [0, 25], [-25, 700]]
    start = time.time()
    figure, ax = plt.subplots(1, 5, figsize=(15, 3))
    figure.suptitle("Conditional GAN (WGAN-GP) on Collision Data")
    sns.set(color_codes=True, style='white', palette='colorblind')
    loss_G_train = []
    loss_D_train = []
    for epoch in range(epochs):
        for data_batch in dataset:
            G_loss = train_G(data_batch)
            for _ in range(D_cycle):
                D_loss = train_D(data_batch)

        loss_G_train.append(G_loss.numpy())
        loss_D_train.append(D_loss.numpy())

        if (epoch + 1) % steps_show == 0:
            num_test = 1000
            condition = collision_data[:num_test, :3]
            noise = tf.random.normal([num_test, noise_dim], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32)
            generated_out = generator([condition, noise], training=True)

            ### Figure Velocity Density ###
            for i in range(3):
                ax[i].clear()
                ax[i].set_ylim(list_lim[i])
                ax[i].set_xlim(list_lim[i])

                plot_data = collision_data.numpy()[:num_test, :]
                ax[i].plot(plot_data[:num_test, i], plot_data[:num_test, i + 3], '.b', alpha=.1)
                plot_data = np.concatenate([condition.numpy(), generated_out.numpy()], axis=1)
                ax[i].plot(plot_data[:num_test, i], plot_data[:num_test, i + 3], '.r', alpha=.5)

            ### Figure Energy Density ###
            i = 3
            ax[i].clear()
            ax[i].set_ylim(list_lim[i])
            ax[i].set_xlim(list_lim[i])

            plot_data = collision_data.numpy()[:num_test, :]
            ax[i].plot(np.sum(plot_data[:num_test, :3] ** 2, axis=1),
                       np.sum(plot_data[:num_test, 3:] ** 2, axis=1), '.b', alpha=.1)
            plot_data = np.concatenate([condition.numpy(), generated_out.numpy()], axis=1)
            ax[i].plot(np.sum(plot_data[:num_test, :3] ** 2, axis=1),
                       np.sum(plot_data[:num_test, 3:] ** 2, axis=1), '.r', alpha=.5)

            ### Figure Discriminator Loss ###
            i = 4
            ax[i].clear()
            ax[i].plot([-i for i in loss_D_train], '-')
            ax[i].set_title('Negative critic loss')
            ax[i].set_xlabel('Epoch')

            ### Flush ###
            figure.canvas.draw()
            figure.canvas.flush_events()
            plt.pause(0.00001)

        if (epoch + 1) % 50 == 0:
            print('Time for epoch {}/{} is {} sec'.format(epoch + 1, epochs, time.time() - start))
            start = time.time()
    figure.show()

    return loss_G_train, loss_D_train, figure

if __name__ == '__main__':
    loss_G_train, loss_D_train, figure = train(train_dataset, epochs=epoch_num, D_cycle=D_cycle, steps_show=steps_show)
    figure.savefig('cWGAN.png')
    print('Training finished, result is saved in cWGAN.png')