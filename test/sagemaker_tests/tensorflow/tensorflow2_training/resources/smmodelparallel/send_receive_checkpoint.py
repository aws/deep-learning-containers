# Standard Library
import filecmp
import os
import shutil
import time

# First Party
import smdistributed.modelparallel.tensorflow as smp
import smdistributed.modelparallel.tensorflow.utils as utils

smp.init({"partitions": 2, "fork": False})


def generate_big_random_bin_file(filename, size):
    """
    generate big binary file with the specified size in bytes
    :param filename: the filename
    :param size: the size in bytes
    :return:void
    """
    with open("%s" % filename, "wb+") as fout:
        fout.write(os.urandom(size))


start_time = time.time()
src_root_dir = "./send_receive_checkpoint_test"
dst_root_dir = "./send_receive_checkpoint_result"
filename = "data.bin"

if smp.rank() != 0:
    file_path = os.path.join(src_root_dir, "mp_rank_" + str(smp.rank()))

    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path, exist_ok=True)
    # creating a 1MB file.
    generate_big_random_bin_file(os.path.join(file_path, filename), 1024 * 1024)

    # sending to rank 0
    utils.send_checkpoint_files(src_root_dir, 0)

else:
    # receving from  rank 1
    utils.receive_checkpoint_files(dst_root_dir, 1)

smp.core.barrier()

if smp.rank() == 0:
    src_file = os.path.join(src_root_dir, "mp_rank_1", filename)
    dst_file = os.path.join(dst_root_dir, "mp_rank_1", filename)

    # comparing the files.
    assert filecmp.cmp(src_file, dst_file), "Source and Destination files do not match."

    # cleanup:
    if os.path.exists(src_root_dir):
        shutil.rmtree(src_root_dir)
    if os.path.exists(dst_root_dir):
        shutil.rmtree(dst_root_dir)
