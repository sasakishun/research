import os

## Network config
##  Input width and Height
#Height, Width = 28, 28  # MNIST
#Height, Width = 64, 64
#Channel = 3

# input data shape
# channels_last -> [mb, c, h, w] , channels_first -> [mb, h, w, c]
Input_type = 'channels_last'

## Directory paths for training
#Train_dirs = [
#    '/mnt/c/Users/demo/Research_nagayosi/Dataset/Moca',
#    '/mnt/c/Users/demo/Research_nagayosi/Dataset/text_image'
#]

## Data augmentation
#Horizontal_flip = True
#Vertical_flip = False
#Rotate_ccw90 = False

File_extensions = ['.jpg', '.png']

## Training config
Iteration = 10000# 0# 00
Minibatch = 64# 32# 64

## Test config
## The total number of generated images is Test_Minibatch * Test_num
Test_Minibatch = 100
Test_num = 10
Save_test_img_dir = 'test_images_mnist'

# if Save_combine is True, generated images in test are stored combined with same minibatch's
# if False, generated images are stored separately
# if None, generated image is not stored
Save_train_combine = True
Save_test_combine = True

Save_train_step = 1000
Save_iteration_disp = True
Shrinked = ""
## Save config
Dataset = ""
Save_dir = 'models_mnist'
Save_d_name = 'D.h5'
Save_g_name = 'G.h5'
Save_c_name = 'C.h5'
Save_classify_name = 'Classify.h5'
Save_freezed_classify_1_name = 'Freezed_Classify_1.h5'
Save_binary_classify_name = 'Binary_classify.h5'
Save_tree_name = 'Tree.h5'
Save_mlp_name = "MLP"+Shrinked+".h5"
Save_syncro_name = 'Syncro_layer.h5'
Save_d_path = os.path.join(Save_dir, Save_d_name)
Save_g_path = os.path.join(Save_dir, Save_g_name)
Save_c_path = os.path.join(Save_dir, Save_c_name)
Save_classify_path = os.path.join(Save_dir, Save_classify_name)
Save_freezed_classify_1_path = os.path.join(Save_dir, Save_freezed_classify_1_name)
Save_binary_classify_path = os.path.join(Save_dir, Save_binary_classify_name)
Save_tree_path = os.path.join(Save_dir, Save_tree_name)
Save_mlp_path = os.path.join(Save_dir, Save_mlp_name)
Save_np_mlp_path = os.path.join(Save_dir, Save_mlp_name[:-3]+".npy")
Save_hidden_layers_path = []
for i in range(5):
    Save_hidden_layers_path.append(os.path.join(Save_dir, Dataset + 'hidden_layers{}.h5'.format(i)))
Save_layer_mask_path = []# 'layer_mask.npy'
for i in range(10):
    Save_layer_mask_path.append(os.path.join(Save_dir, Dataset + 'layer_mask{}.npy'.format(i)))
Save_syncro_path = os.path.join(Save_dir, Save_syncro_name)
Save_train_img_dir = 'train_images_mnist'
Save_img_num = 5
def reload_path():
    global Dataset
    global Save_d_name
    global Save_g_name
    global Save_c_name
    global Save_classify_name
    global Save_freezed_classify_1_name
    global Save_binary_classify_name
    global Save_tree_name
    global Save_syncro_name
    global Save_mlp_name

    global Save_d_path
    global Save_g_path
    global Save_c_path
    global Save_classify_path
    global Save_freezed_classify_1_path
    global Save_binary_classify_path
    global Save_tree_path
    global Save_mlp_path
    global Save_np_mlp_path
    global Save_shrinked_mlp_path
    global Save_shrinked_np_mlp_path

    global  Shrinked
    Save_d_name = Dataset + 'D.h5'
    Save_g_name = Dataset + 'G.h5'
    Save_c_name = Dataset + 'C.h5'
    Save_classify_name = Dataset + 'Classify.h5'
    Save_freezed_classify_1_name = Dataset + 'Freezed_Classify_1.h5'
    Save_binary_classify_name = Dataset + 'Binary_classify.h5'
    Save_tree_name = Dataset + 'Tree.h5'
    Save_syncro_name = Dataset + 'Syncro_layer.h5'
    Save_mlp_name = Dataset + "MLP"+Shrinked+".h5"

    Save_d_path = os.path.join(Save_dir, Save_d_name)
    Save_g_path = os.path.join(Save_dir, Save_g_name)
    Save_c_path = os.path.join(Save_dir, Save_c_name)
    Save_classify_path = os.path.join(Save_dir, Save_classify_name)
    Save_freezed_classify_1_path = os.path.join(Save_dir, Save_freezed_classify_1_name)
    Save_binary_classify_path = os.path.join(Save_dir, Save_binary_classify_name)
    Save_tree_path = os.path.join(Save_dir, Save_tree_name)
    Save_mlp_path = os.path.join(Save_dir, Save_mlp_name)
    Save_np_mlp_path = os.path.join(Save_dir, Save_mlp_name[:-3] + ".npy")
    return

## Other config
## Randon_seed is used for seed of dataset shuffle in data_loader.py
Random_seed = 0

## Check
variety = ['channels_first', 'channels_last']
if not Input_type in variety:
    raise Exception("unvalid Input_type")

#os.system("rm {}/*".format(Save_train_img_dir))
import platform
python_version = int(platform.python_version_tuple()[0])
if python_version == 3:
    os.makedirs(Save_dir, exist_ok=True)
    os.makedirs(Save_train_img_dir, exist_ok=True)
    os.makedirs(Save_test_img_dir, exist_ok=True)
elif python_version == 2:
    if not os.path.exists(Save_dir):
        os.makedirs(Save_dir)
    if not os.path.exists(Save_train_img_dir):
        os.makedirs(Save_train_img_dir)
    if not os.path.exists(Save_test_img_dir):
        os.makedirs(Save_test_img_dir)
