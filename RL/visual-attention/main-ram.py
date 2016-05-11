import common.mnist_loader as mnist_loader
import numpy
import time
import math

dataset = mnist_loader.read_data_sets("data")
save_dir = "model/"
save_prefix = "save"
start_step = 10000
load_path = None
# load_path = save_dir + save_prefix + str(start_step) + '.h5'

# To enable visualization, set draw to True
eval_only = False
animate = True
draw = True

minRadius = 4
sensorBandwidth = 8  # fixed resolution of sensor
sensorArea = sensorBandwidth ** 2
depth = 3  # channels of zoom
channels = 1  # grayscale image
totalSensorBandwidth = depth * sensorBandwidth * \
                       sensorBandwidth * channels
batch_size = 10

hg_size = 128
hl_size = 128
g_size = 256
cell_size = 256
cell_out_size = cell_size

glimpses = 6
n_classes = 10

lr = 1e-3
max_iters = 1000000

mnist_size = 28

loc_sd = 0.1
mean_locs = []
sampled_locs = []  # ~N(mean_locs[.], loc_sd)
glimpse_images = []  # to show in window


