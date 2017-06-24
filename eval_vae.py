from libs.vae_face import train_vae_face
from libs.vae import train_vae
from libs.vae import train_vae_align
from libs.vae import eval_vae
from libs.vae import eval_vae_align
import tensorflow as tf
import os

def test_align(path):
    fs = [os.path.join(path, f)
              for f in os.listdir(path) if f.endswith('.png')]
    #print(fs)
    eval_vae_align(
        files=fs,
        input_shape=[128, 128, 1],
        batch_size=69,
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
       ckpt_name='models_bak/align-300w-gtbbx-1229-02')
        
if __name__ == '__main__':
    test_align('./')
