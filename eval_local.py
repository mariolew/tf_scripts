from libs.vae_face import train_vae_face
from libs.vae import train_vae
from libs.vae import train_vae_align
from libs.vae import eval_vae
from libs.vae import eval_local
import tensorflow as tf
import os

def test_align(path):
    fs = [os.path.join(path, f)
              for f in os.listdir(path) if f.endswith('.png')]
    #print(fs)
    eval_local(
        files=fs,
        input_shape=[128, 128, 1],
        batch_size=1,
        n_epochs=1,
        crop_shape=[128, 128, 1],
        crop_factor=1,
        convolutional=True,
        variational=True,
        n_filters=[100, 100, 100],
        n_hidden=250,
        n_code=100,
        dropout=False,
        keep_prob=1.0,
        filter_sizes=[3, 3, 3],
        activation=tf.nn.relu,
       ckpt_name='models_bak/vae_gt20000-20000')
       # ckpt_name='./300w_vae-0')
if __name__ == '__main__':
    test_align('./')
