import json
import os
import horovod.tensorflow as hvd

hvd.init()

with open(os.path.join('/opt/ml/model/local-rank-%s-rank-%s' % (hvd.local_rank(), hvd.rank())), 'w+') as f:
    basic_info = {'local-rank': hvd.local_rank(), 'rank': hvd.rank(), 'size': hvd.size()}

    print(basic_info)
    json.dump(basic_info, f)
