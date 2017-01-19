import tensorflow as tf
import numpy as np


class GridPrediction:
    def __init__(self, obs_dim, graph, normalizer=1., num_cuts=1):
        self.normal_factor = normalizer
        self.num_cuts = num_cuts
        self.cut_width = obs_dim/num_cuts
        self.num_boxes = self.cut_width**2
        with graph.as_default():
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, 2*self.num_boxes])


        pass


    def calculate_batch_intensity(self, im_batch):
        batch_im_arr = np.zeros(shape=[len(im_batch), self.num_cuts, self.num_cuts])
        for k in range(len(im_batch)):
            for i in range(self.num_cuts):
                for j in range(self.num_cuts):
                    batch_im_arr[k][i][j] = im_batch[k, self.cut_width*i:self.cut_width*i, self.cut_width*j:self.cut_width*j]

        return batch_im_arr

    def split_image(self, im):
        cells = np.reshape(im, newshape=[self.num_cuts, self.num_cuts, self.cut_width, self.cut_width, 3])
        return cells


    def average_absolute_intensity_change(self, cell1, cell2):
        return np.average(np.abs(cell1 - cell2))



    def calculate_intensity_change(self, im1, im2):
        cells1 = self.split_image(im1)
        cells2 = self.split_image(im2)
        intensities = np.zeros(shape=[self.num_cuts, self.num_cuts])
        for i in range(self.num_cuts):
            for j in range(self.num_cuts):
                intensities[i][j] = self.average_absolute_intensity_change(cells1[i][j], cells2[i][j])


        return intensities






class ProjectionQuestionNet:
    def __init__(self):
        pass


class FeatureQuestionNet:
    def __init__(self):
        pass

