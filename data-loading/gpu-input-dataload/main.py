import types
import collections
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import matplotlib.pyplot as plt
from dataload_helper import ExternalInputGpuIterator

batch_size = 16


eii_gpu = ExternalInputGpuIterator(batch_size)

print(type(next(iter(eii_gpu))[0][0]))


pipe_gpu = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
with pipe_gpu:
    images, labels = fn.external_source(source=eii_gpu, num_outputs=2, device="gpu", dtype=types.UINT8)
    enhance = fn.brightness_contrast(images, contrast=2)
    pipe_gpu.set_outputs(enhance, labels)

pipe_gpu.build()

pipe_out_gpu = pipe_gpu.run()
batch_gpu = pipe_out_gpu[0].as_cpu()
labels_gpu = pipe_out_gpu[1].as_cpu()

img = batch_gpu.at(2)
print(img.shape)
print(labels_gpu.at(2))
plt.axis('off')
plt.imshow(img)
plt.show()
