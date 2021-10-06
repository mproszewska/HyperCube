import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import h5py
import cv2
import argparse
from ops import *
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-c", type=str)
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)

args = parser.parse_args()
data = torch.load(args.input).numpy()
c = args.c

c = args.c
if c == "airplane":
    NUM_SAMPLES = 809
elif c == "car":
    NUM_SAMPLES = 1500
elif c == "chair":
    NUM_SAMPLES = 1356
elif c == "rifle":
    NUM_SAMPLES = 475
elif c == "table":
    NUM_SAMPLES = 1702
else:
    raise NotImplemented


if not os.path.exists(args.output):
    os.makedirs(args.output)
if not os.path.exists(args.output + "/checkpoint"):
    os.makedirs(args.output + "/checkpoint")


run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
log_device_placement = True


class ZGAN(object):
    def __init__(
        self,
        sess,
        is_training=False,
        z_vector_dim=256,
        z_dim=256,
        df_dim=2048,
        gf_dim=2048,
        dataset_name="default",
        checkpoint_dir=None,
        sample_dir=None,
    ):
        self.sess = sess

        self.z_dim = z_dim
        self.z_vector_dim = z_vector_dim

        self.df_dim = df_dim
        self.gf_dim = gf_dim

        self.checkpoint_dir = checkpoint_dir

        self.data_z = data

        self.build_model()

    def build_model(self):
        self.z_vector = tf.placeholder(
            shape=[None, self.z_vector_dim], dtype=tf.float32
        )
        self.z = tf.placeholder(shape=[None, self.z_dim], dtype=tf.float32)

        self.G = self.generator(self.z, reuse=False)
        self.D = self.discriminator(self.z_vector, reuse=False)
        self.D_ = self.discriminator(self.G, reuse=True)

        self.sG = self.generator(self.z, reuse=True)

        self.d_loss = tf.reduce_mean(self.D) - tf.reduce_mean(self.D_)
        self.g_loss = tf.reduce_mean(self.D_)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.z_vector + (1 - epsilon) * self.G
        d_hat = self.discriminator(x_hat, reuse=True)

        ddx = tf.gradients(d_hat, x_hat)[0]
        print(ddx.get_shape().as_list())
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0)

        self.d_loss = self.d_loss + ddx

        self.vars = tf.trainable_variables()
        self.g_vars = [var for var in self.vars if "g_" in var.name]
        self.d_vars = [var for var in self.vars if "d_" in var.name]

        self.saver = tf.train.Saver(max_to_keep=20)

    def generator(self, z, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            h1 = lrelu(linear(z, self.gf_dim, "g_1_lin"))
            h2 = lrelu(linear(h1, self.gf_dim, "g_2_lin"))
            h3 = linear(h2, self.z_vector_dim, "g_3_lin")
            return tf.nn.sigmoid(h3)

    def discriminator(self, z_vector, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h1 = lrelu(linear(z_vector, self.df_dim, "d_1_lin"))
            h2 = lrelu(linear(h1, self.df_dim, "d_2_lin"))
            h3 = linear(h2, 1, "d_3_lin")
            return h3

    def train(self, num_epochs=10000, learning_rate=0.00005, beta1=0.5):
        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
            self.d_loss, var_list=self.d_vars
        )
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
            self.g_loss, var_list=self.g_vars
        )
        self.sess.run(tf.global_variables_initializer())

        counter = 0
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter + 1
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        batch_index_num = len(self.data_z)
        batch_index_list = np.arange(batch_index_num)
        batch_size = 50
        batch_num = int(batch_index_num / batch_size)

        for epoch in range(counter, num_epochs + 1):
            np.random.shuffle(batch_index_list)
            errD_total = 0
            errG_total = 0
            for minib in range(batch_num):
                batch_z = np.random.normal(0, 0.2, [batch_size, self.z_dim]).astype(
                    np.float32
                )
                batch_vector_z = self.data_z[
                    minib * batch_size : (minib + 1) * batch_size
                ]

                # Update D network
                _, errD = self.sess.run(
                    [d_optim, self.d_loss],
                    feed_dict={self.z_vector: batch_vector_z, self.z: batch_z},
                )
                # Update G network
                _, errG = self.sess.run(
                    [g_optim, self.g_loss], feed_dict={self.z: batch_z}
                )
                errD_total += errD
                errG_total += errG
            if epoch % 10 == 0:
                print(
                    "Epoch: [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                    % (
                        epoch,
                        num_epochs,
                        time.time() - start_time,
                        errD_total / batch_num,
                        errG_total / batch_num,
                    )
                )

            if epoch % 500 == 0:
                self.save(self.checkpoint_dir, epoch)

                # training z
                z_height = 64
                z_counter = np.zeros([z_height, self.z_vector_dim], np.int32)
                z_img = np.zeros([z_height, self.z_vector_dim], np.uint8)

                z_vector = self.data_z
                for i in range(batch_index_num):
                    for j in range(self.z_dim):
                        slot = int(z_vector[i, j] * (z_height - 0.0001))
                        if slot > z_height or slot < 0:
                            print("error slot")
                        z_counter[slot, j] += 1

                maxz = 50  # np.max(z_counter)
                for i in range(z_height):
                    for j in range(self.z_dim):
                        x = int(z_counter[i, j] * 256 / maxz)
                        if x > 255:
                            x = 255
                        z_img[i, j] = x

                cv2.imwrite("z_train.png", z_img)

                # generated z
                z_height = 64
                z_counter = np.zeros([z_height, self.z_vector_dim], np.int32)
                z_img = np.zeros([z_height, self.z_vector_dim], np.uint8)

                batch_z = np.random.normal(
                    0, 0.2, [batch_index_num, self.z_dim]
                ).astype(np.float32)
                z_vector = self.sess.run(self.sG, feed_dict={self.z: batch_z})
                for i in range(batch_index_num):
                    for j in range(self.z_dim):
                        slot = int(z_vector[i, j] * (z_height - 0.0001))
                        if slot > z_height or slot < 0:
                            print("error slot")
                        z_counter[slot, j] += 1

                for i in range(z_height):
                    for j in range(self.z_dim):
                        x = int(z_counter[i, j] * 256 / maxz)
                        if x > 255:
                            x = 255
                        z_img[i, j] = x

                cv2.imwrite("z_gen.png", z_img)

                print("[Visualized Z]")

    def get_z(self, num, interpolate=False):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        # generated z
        batch_z = np.random.normal(0, 0.2, [num, self.z_dim]).astype(np.float32)
        if interpolate:
            steps = 8
            bz = np.zeros([100 * steps, self.z_dim])
            for i in range(100):
                for j in range(steps):
                    b = float(j) / (steps - 1)
                    bz[i * steps + j] = (1 - b) * batch_z[i] + b * batch_z[i + 1]
            batch_z = bz
        z_vector = self.sess.run(self.sG, feed_dict={self.z: batch_z})
        return z_vector

    @property
    def model_dir(self):
        return "{}_{}_{}".format(args.c, self.z_dim, self.z_vector_dim)

    def save(self, checkpoint_dir, step):
        model_name = "ZGAN.model"
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(
            self.sess, os.path.join(checkpoint_dir, model_name), global_step=step
        )

    def load(self, checkpoint_dir):
        import re

        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


with tf.Session(config=run_config) as sess_z:
    zgan = ZGAN(
        sess_z,
        is_training=True,
        checkpoint_dir=args.output + "/checkpoint",
        sample_dir=args.output + "/samples",
    )
    zgan.train()
tf.reset_default_graph()

with tf.Session(config=run_config) as sess_z:
    zgan = ZGAN(sess_z, is_training=False, checkpoint_dir=args.output + "/checkpoint")
    generated_z = zgan.get_z(NUM_SAMPLES * 5)
    print("output shape", generated_z.shape)
    torch.save(torch.from_numpy(generated_z), args.output + "/gen_latent.pt")
tf.reset_default_graph()
