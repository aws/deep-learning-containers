import json
import os
import horovod.tensorflow as hvd

hvd.init()

with open('/opt/ml/model/local-rank-%s-rank-%s' % (hvd.local_rank(), hvd.rank()), 'w+') as f:
    basic_info = {'local-rank': hvd.local_rank(), 'rank': hvd.rank(), 'size': hvd.size()}

    print(basic_info)
    json.dump(basic_info, f)

val = os.environ.get('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI')
host = os.environ.get('SM_CURRENT_HOST')

assert val is not None
assert host is not None

print('host {}: AWS_CONTAINER_CREDENTIALS_RELATIVE_URI={}'.format(host, val))
