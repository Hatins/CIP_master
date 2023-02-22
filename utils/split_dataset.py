# ------------------------------------------------------------------------
# CIP: Categorical Inference Poisoning: Verifiable Defense Against Black-Box DNN Model Stealing Without Constraining Surrogate Data and Query Times
# Haitian Zhang, Guang Hua, Xinya Wang, Hao Jiang, and Wen Yang
# paper: https://ieeexplore.ieee.org/document/10042038
# -----------------------------------------------------------------------

import os, random, shutil, sys

def create_N_dir(path, N):
    dir_length = N
    for i in range(dir_length):
        os.mkdir(path + '/' + str(i))
        print('{} th file has been created'.format(i))


def rename_with_index(path,length = 0):
    old_names = os.listdir(path)
    for index, old_name in enumerate(old_names):
        if old_name != sys.argv[0]:
            # os.rename(os.path.join(path, old_name), os.path.join(path, 'N{}.png'.format(index + 1 )))  # unsuccessive
            # print(old_name, "has been renamed successfully! New name is: N{}.png".format(index + 1))

            os.rename(os.path.join(path, old_name),os.path.join(path, 'N{}.png'.format(index + 1+length )))  # successive
            # print(old_name, "has been renamed successfully! New name is: N{}.png".format(index + 1+length ))
    length = length + len(old_names)
    return length

#type = 0 move file; type = 1 copy file
def moveFile(fileDir, tarDir, type = 0):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    rate = 0.5
    picknumber = int(filenumber * rate)
    sample = random.sample(pathDir, picknumber)
    if type == 0:
        for name in sample:
            shutil.move(fileDir + name, tarDir + name)
            # print('successful move to ' + tarDir)
    else:
        for name in sample:
            shutil.copy(fileDir + name, tarDir + name)
            # print('successful copy to ' + tarDir)
    return

# file structure:
#--Dataset
#----MNIST

# return both MNIST_half and MNIST_surrogate with half of training images
#--Dataset
#----MNIST_half (with labels)
#----MNIST_surrogate (without labels)

def split_dataset(path, dataset_name):
    length = 0
    dataset_path = path + '/' +  dataset_name
    dataset_path_copy = dataset_path + '_' + 'half'
    shutil.copytree(dataset_path, dataset_path_copy)
    surrogate_dataset_path = path + '/' + dataset_name + '_' + 'surrogate'
    os.mkdir(surrogate_dataset_path)
    fileDir = dataset_path_copy + '/' + 'train'
    file_address = os.listdir(fileDir)
    # The surrogate portion is without label, so files therein are renamed to avoid name conflict
    for item in file_address:
        length = rename_with_index(fileDir +'/'+ item, length)
    tarDir = surrogate_dataset_path
    file_address = os.listdir(fileDir)
    for item in file_address:
        file_inputs = fileDir +r'/{}'.format(item)+'/'
        file_outputs = tarDir +'/'
        moveFile(file_inputs, file_outputs)

if __name__ == '__main__':
    path = './dataset'
    dataset_name = 'CIFAR10'
    split_dataset(path, dataset_name)

