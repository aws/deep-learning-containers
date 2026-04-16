set -ex

export SM_HP_MP_PARAMETERS=\{\"ddp\":true,\"microbatches\":4,\"partitions\":2,\"pipeline\":\"interleaved\"\}
mpirun -mca btl_vader_single_copy_mechanism none --allow-run-as-root -np 8 python smmodelparallel_pt_mnist.py --assert-losses 1 --data-dir data/training