import cupy as cp
import imageio

class ExternalInputGpuIterator(object):
    def __init__(self, batch_size):
        self.images_dir = "/workdir/dali-data-loading/data/images/"
        self.batch_size = batch_size
        with open(self.images_dir + "file_list.txt", 'r') as f:
            self.files = [line.rstrip() for line in f if line != '']
        shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            jpeg_filename, label = self.files[self.i].split(' ')
            im = imageio.imread(self.images_dir + jpeg_filename)
            im = cp.asarray(im)
            im = im * 0.6;
            batch.append(im.astype(cp.uint8))
            labels.append(cp.array([label], dtype = np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)
