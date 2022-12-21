import os

from Config import cache_folder


def exist_cache(file_name):
    return os.path.exists(cache_folder + file_name)


def save_cache(data, file_name):
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    f = open(cache_folder + file_name, 'w')
    for u in range(len(data)):
        f.write(str(data[u]))
        if u != len(data) - 1:
            f.write(',')
    f.flush()
    f.close()


def read_cache(file_name):
    if os.path.exists(cache_folder + file_name):
        f = open(cache_folder + file_name, 'r')
        return f.read().split(',')
    else:
        assert False, 'Cache not found'
