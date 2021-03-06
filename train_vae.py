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
    train_vae_align(
        files=fs,
        input_shape=[128, 128, 1],
        batch_size=64,
        n_epochs=50,
        crop_shape=[128, 128, 1],
        crop_factor=1,
        convolutional=True,
        variational=True,
        n_filters=[100, 100, 100],
        n_hidden=250,
        n_code=100,
        dropout=True,
        filter_sizes=[3, 3, 3],
        activation=tf.nn.relu,
        #ckpt_name = '/mnt/dataset2/tea/tfmodels/align-300w-gtbbx-1218')
        ckpt_name = 'models/vae_gt20000-20000')
        #ckpt_name='/mnt/dataset2/tea/tfmodels/align-300w-426')
        
if __name__ == '__main__':
    test_align('./')
    #train_vae_face()
