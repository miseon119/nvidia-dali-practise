import types
import collections
import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import matplotlib.pyplot as plt
from dataload_helper import ExternalInputIterator

batch_size = 16

# Define the Pipeline
eii = ExternalInputIterator(batch_size)

pipe = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
with pipe:
    jpegs, labels = fn.external_source(source=eii, num_outputs=2, dtype=types.UINT8)
    decode = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    enhance = fn.brightness_contrast(decode, contrast=2)
    pipe.set_outputs(enhance, labels)

pipe.build()
pipe_out = pipe.run()

batch_cpu = pipe_out[0].as_cpu()
labels_cpu = pipe_out[1]

img = batch_cpu.at(2)
print(img.shape)
print(labels_cpu.at(2))
plt.axis('off')
plt.imshow(img)
plt.show()
