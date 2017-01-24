import numpy as np



def check_valid_split(im, num_cuts):
    rem1 = len(im[0]) % num_cuts
    rem2 = len(im[1]) % num_cuts
    if rem1+rem2 == 0:
        return True
    else:
        return False


def split_image(im, num_cuts):
    cells = np.reshape(im, newshape=[num_cuts, num_cuts, len(im[0])/num_cuts, len(im[1])/num_cuts, -1])
    return cells


def average_absolute_intensity_change(cell1, cell2):
    return np.average(np.abs(cell1 - cell2))



def calculate_intensity_change(im1, im2, num_cuts):
    cells1 = split_image(im1, num_cuts)
    cells2 = split_image(im2, num_cuts)
    intensities = np.zeros(shape=[num_cuts, num_cuts])
    for i in range(num_cuts):
        for j in range(num_cuts):
            intensities[i][j] = average_absolute_intensity_change(cells1[i][j], cells2[i][j])

    return intensities
