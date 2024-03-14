from scipy.ndimage import gaussian_filter
import tifffile
import numpy as np
import napari
from organoid.preprocessing._smoothing import _gaussian_smooth
#import otsu
from skimage.filters import threshold_otsu, threshold_local
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from functools import partial
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from organoid.preprocessing._smoothing import _smooth_gaussian


# path_to_data = '/home/jvanaret/data/project_egg/raw/fusion4'
# data = tifffile.imread(f'{path_to_data}/fusion4.tif')[7]

path_to_data = '/home/jvanaret/data/new_fusions'
image = tifffile.imread(f'{path_to_data}/delme_1_registered_rigid3D-183-87.tif')[7]

###
sigma_blur = 6
threshold_factor = 1
###


percs = np.percentile(image, [1, 99])
image = (image - percs[0]) / (percs[1] - percs[0])
image = np.clip(image, 0, 1)

nonzero_mask = image > 0

if np.any(nonzero_mask):
    blurred = _smooth_gaussian(image, mask=nonzero_mask, sigmas=sigma_blur)
    blurred2 = _smooth_gaussian(image**2, mask=nonzero_mask, sigmas=sigma_blur)

    sigma = blurred2 - blurred**2

    snp_array = sigma * blurred
    snp_mask = snp_array > 0
    # snp_array = np.log(snp_array, where=(snp_array != 0))

    snp_array = np.log(
        snp_array, 
        where=np.logical_and(nonzero_mask, snp_mask)
    )
else:
    blurred = _smooth_gaussian(image, sigmas=sigma_blur)
    blurred2 = _smooth_gaussian(image**2, sigmas=sigma_blur)

    sigma = blurred2 - blurred**2

    snp_array = sigma * blurred
    snp_mask = snp_array > 0
    # snp_array = np.log(snp_array, where=(snp_array != 0))

    snp_array = np.log(
        snp_array, 
        where=snp_mask
    )

threshold = threshold_otsu(snp_array[snp_mask]) * threshold_factor

# Create a binary mask
# binary_mask = snp_array > threshold
binary_mask = np.logical_and(snp_array > threshold, snp_mask)


viewer = napari.Viewer()
viewer.add_image(image, name='data')
viewer.add_image(nonzero_mask, name='nonzero_mask')
viewer.add_image(sigma, name='sigma')
viewer.add_image(snp_mask, name='snp_mask')
viewer.add_image(snp_array, name='snp_array')
viewer.add_image(binary_mask, name='mask')
# data = data/data.max()

# percs = np.percentile(data, [1, 99])
# data = (data - percs[0]) / (percs[1] - percs[0])
# data = np.clip(data, 0, 1)



# blur = gaussian_filter(data, sigma=10)
# blur2 = gaussian_filter(data**2, sigma=10)

# sigma = np.sqrt(np.clip(blur2 - blur**2, 0, None))

# im1 = sigma * blur
# min_val = np.percentile(im1[im1!=0], 1)
# im1 = np.where(im1==0, min_val, im1)
# logim = np.log(im1)

# percs = np.percentile(logim, [1, 99])
# # logim = (logim -logim.min()) / (logim.max() - logim.min()) 
# # logim = (logim - percs[0]) / (percs[1] - percs[0])
# # logim = np.clip(logim, 0, 1)


# # print((threshold_otsu(logim) - logim.min())/(logim.max() - logim.min()))
# # print((threshold_otsu(im1) - im1.min())/(im1.max() - im1.min()))

# mask = logim > threshold_otsu(logim)
# # mask2 = im1 > threshold_otsu(im1)

# fig = plt.figure()
# plt.hist(logim.ravel(), bins=128, )
# plt.yscale('log')
# plt.axvline(threshold_otsu(logim), color='r')
# plt.show()


# viewer.add_image(mask2, name='mask2')




# func = partial(zoom, zoom=0.5, order=1, mode='reflect', prefilter=False)
# data = np.array(
#     process_map(
#         func, data, max_workers=7, chunksize=1
#     )
# )

# func = partial(threshold_local, block_size=7)
#     # mask = np.array(
#     #     [threshold_local(data[i], block_size=3, ) for i in tqdm(range(data.shape[0]))]
#     # )
# mask = data>np.array(
#     process_map(
#         func, data, max_workers=7, chunksize=1)
# )


# percs = np.percentile(data, [1, 99])
# data = (data - percs[0]) / (percs[1] - percs[0])
# data = np.clip(data, 0, 1)


# blur = _gaussian_smooth(data, sigmas=2)
# blur2 = _gaussian_smooth(data**2, sigmas=2)

# sigma = np.sqrt(np.clip(blur2 - blur**2, 0, None))

# im1 = sigma * blur
# im2 = blur/sigma

# mask = im2 < threshold_otsu(im2)
# mask2 = im1 > threshold_otsu(im1)
# mask3 = sigma > threshold_otsu(sigma)
# mask4 = blur > threshold_otsu(blur)


# # blur = gaussian_filter(data, sigma=6)
# # blur2 = gaussian_filter(data**2, sigma=6)
# # blur = _gaussian_smooth(data, sigmas=2)
# # blur2 = _gaussian_smooth(data**2, sigmas=2)

# # sigma = np.sqrt(np.clip(blur2 - blur**2, 0, None))

# # im1 = sigma * blur
# # im2 = blur/sigma

# # mask = im2 < threshold_otsu(im2)

# # viewer = napari.Viewer()

# viewer.add_image(data, name='data')
# viewer.add_image(blur)
# viewer.add_image(blur2)
# viewer.add_image(sigma, name='sigma')
# viewer.add_image(im1, name='im1')
# viewer.add_image(im2, name='im2')
# viewer.add_image(mask, name='mask')
# viewer.add_image(mask2, name='mask2')
# viewer.add_image(mask3, name='mask3')
# viewer.add_image(mask4, name='mask4')

napari.run()



