# Import ImageIO
import imageio

# Load "chest-220.dcm"
im = imageio.imread("chest-220.dcm")

# Print image attributes
print('Image type:', type(im))
print('Shape of image array:', im.shape)

# Import ImageIO
import imageio
im = imageio.imread('chest-220.dcm')

# Print the available metadata fields
print(im.meta.keys())
print(im.meta['PatientSex'])




# Import ImageIO and PyPlot
import imageio
import matplotlib.pyplot as plt

# Read in "chest-220.dcm"
im = imageio.imread("chest-220.dcm")
# Draw the image in grayscale
plt.imshow(im, cmap="gray")

# Render the image
plt.show()

# Draw the image with greater contrast
plt.imshow(im, cmap="gray", vmin=-200, vmax=200)

# Render the image
plt.show()

# Remove axis ticks and labels
plt.axis('off')

# Render the image
plt.show()

###################################
# N-dimensional images
###################################
# Import ImageIO and NumPy
import imageio
import numpy as np

# Read in each 2D image
im1 = imageio.imread('chest-220.dcm')
im2 = imageio.imread('chest-221.dcm')
im3 = imageio.imread('chest-222.dcm')

# Stack images into a volume
vol = np.stack([im1, im2, im3])
print('Volume dimensions:', vol.shape)

'''Load volumes
ImageIO's volread() function can load multi-dimensional datasets and create 3D volumes from a folder of images. It can also aggregate metadata across these multiple images.
For this exercise, read in an entire volume of brain data from the "tcia-chest-ct" folder, which contains 25 DICOM images.'''

# Import ImageIO
import imageio

# Load the "tcia-chest-ct" directory
vol = imageio.volread("tcia-chest-ct")

# Print image attributes
print('Available metadata:', vol.meta.keys())
print('Shape of image array:', vol.shape)




"""Field of view
The amount of physical space covered by an image is its field of view, which is calculated from two properties:

Array shape, the number of data elements on each axis. Can be accessed with the shape attribute.
Sampling resolution, the amount of physical space covered by each pixel. Sometimes available in metadata (e.g., meta['sampling']).
For this exercise, multiply the array shape and sampling resolution along each axis to calculate the field of view of vol. All values are in millimeters."""
vol.shape * vol.meta.keys()

"""Generate subplots
You can draw multiple images in one figure to explore data quickly. Use plt.subplots() to generate an array of subplots.

fig, axes = plt.subplots(nrows=2, ncols=2)


To draw an image on a subplot, call the plotting method directly from the subplot object rather than through PyPlot: axes[0,0].imshow(im) rather than plt.imshow(im).

For this exercise, draw im1 and im2 on separate subplots within the same figure."""

# Import PyPlot
import matplotlib.pyplot as plt

# Initialize figure and axes grid
fig, axes = plt.subplots(nrows=2, ncols=1)

# Draw an image on each subplot
axes[0].imshow(im1, cmap='gray')
axes[1].imshow(im2, cmap='gray')

# Remove ticks/labels and render
axes[0].axis('off')
axes[1].axis('off')
plt.show()

"""Slice 3D images
The simplest way to plot 3D and 4D images by slicing them into many 2D frames. Plotting many slices sequentially can create a "fly-through" effect that helps you understand the image as a whole.



To select a 2D frame, pick a frame for the first axis and select all data from the remaining two: vol[0, :, :]

For this exercise, use for loop to plot every 40th slice of vol on a separate subplot. matplotlib.pyplot (as plt) has been imported for you.

Instructions
100 XP
Using plt.subplots(), initialize a subplots grid with 1 row and 4 columns.
Plot every 40th slice of vol in grayscale. To get the appropriate index, multiply ii by 40.
Turn off the ticks, labels, and frame for each subplot.
Render the figure.

"""
# Plot the images on a subplots array
fig, axes = plt.subplots(nrows=1, ncols=4)

# Loop through subplots and draw image
for ii in range(4):
    im = vol[ii * 40, :, :]
    axes[ii].imshow(im, cmap='gray')
    axes[ii].axis("off")

# Render the figure
plt.show()


'''Plot other views
Any two dimensions of an array can form an image, and slicing along different axes can provide a useful perspective. However, unequal sampling rates can create distorted images.



Changing the aspect ratio can address this by increasing the width of one of the dimensions.

For this exercise, plot images that slice along the second and third dimensions of vol. Explicitly set the aspect ratio to generate undistorted images.

Instructions
100 XP
Slice a 2D plane from vol where "axis 1" is 256.
Slice a 2D plane from vol where "axis 2" is 256.
For each image, calculate the aspect ratio by dividing the image "sampling" rate for axis 0 by its opponent axis. This information is in vol.meta.
Plot the images in a subplots array. Specify the aspect ratio for each image, and set cmap='gray'.

'''
# Select frame from "vol"
im1 = vol[:, 256, :]
im2 = vol[:, :, 256]

# Compute aspect ratios
d0, d1, d2 = vol.meta['sampling']
asp1 = d0 / d2
asp2 = d0 / d1

# Plot the images on a subplots array
fig, axes = plt.subplots(nrows=2, ncols=1)
axes[0].imshow(im1, cmap='gray', aspect=asp1)
axes[1].imshow(im2, cmap='gray', aspect=asp2)

plt.show()


###################################
# 2- Masks and Filters
###################################
# Cut image processing to the bone by transforming x-ray images. You'll learn how to exploit intensity patterns to select sub-regions of an array, and you'll use convolutional filters to detect interesting features. You'll also use SciPy's ndimage module, which contains a treasure trove of image processing tools.

"""Intensity
In this chapter, we will work with a hand radiograph from a 2017 Radiological Society of North America competition. X-ray absorption is highest in dense tissue such as bone, so the resulting intensities should be high. Consequently, images like this can be used to predict "bone age" in children.

To start, let's load the image and check its intensity range.

The image datatype determines the range of possible intensities: e.g., 8-bit unsigned integers (uint8) can take values in the range of 0 to 255. A colorbar can be helpful for connecting these values to the visualized image.

All exercises in this chapter have the following imports:

import imageio
import numpy as np
import matplotlib.pyplot as plt"""

# Load the hand radiograph
im = imageio.imread("hand-xray.jpg")
print('Data type:', im.dtype)
print('Min. value:', im.min())
print('Max value:', im.max())

# Plot the grayscale image
plt.imshow(im, vmin=0, vmax=255)
plt.colorbar()
format_and_render_plot()



'''Histograms
Histograms display the distribution of values in your image by binning each element by its intensity then measuring the size of each bin.

The area under a histogram is called the cumulative distribution function. It measures the frequency with which a given range of pixel intensities occurs.

For this exercise, describe the intensity distribution in im by calculating the histogram and cumulative distribution function and displaying them together.

Instructions
100 XP
Import scipy.ndimage as ndi.
Generate a 256-bin histogram of im which covers the full range of np.uint8 values.
Calculate the cumulative distribution function for im. First, find the cumulative sum of hist, then divide by the total number of pixels in hist.
Plot hist and cdf on separate subplots. This has been done for you.'''
# Import SciPy's "ndimage" module
import scipy.ndimage as ndi

# Create a histogram, binned at each possible value
hist = ndi.histogram(im, min=0, max=255, bins=256)

# Create a cumulative distribution function
cdf = hist.cumsum() / hist.sum()

# Plot the histogram and CDF
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(hist, label='Histogram')
axes[1].plot(cdf, label='CDF')
format_and_render_plot()

"""Create a mask
Masks are the primary method for removing or selecting specific parts of an image. They are binary arrays that indicate whether a value should be included in an analysis. Typically, masks are created by applying one or more logical operations to an image.

For this exercise, try to use a simple intensity threshold to differentiate between skin and bone in the hand radiograph. (im has been equalized to utilize the whole intensity range.)

Below is the histogram of im colored by the segments we will plot.

Histogram of equalized foot x-ray

Instructions
100 XP
Create a bone mask by selecting pixels with intensities greater than or equal to 145.
Create a skin mask by selecting pixels with intensities greater than or equal to 45 and less than 145.
Plot the skin and bone masks in grayscale."""

# Create skin and bone masks
mask_bone = im >= 145
mask_skin = (im >= 45) & (im < 145)

# Plot the skin (0) and bone (1) masks
fig, axes = plt.subplots(1,2)
axes[0].imshow(mask_skin, cmap='gray')
axes[1].imshow(mask_bone, cmap='gray')
format_and_render_plot()

'''Apply a mask
Although masks are binary, they can be applied to images to filter out pixels where the mask is False.

NumPy's where() function is a flexible way of applying masks. It takes three arguments:

np.where(condition, x, y)
condition, x and y can be either arrays or single values. This allows you to pass through original image values while setting masked values to 0.

Let's practice applying masks by selecting the bone-like pixels from the hand x-ray (im).

Instructions
100 XP
Create a Boolean bone mask by selecting pixels greater than or equal to 145.
Apply the mask to your image using np.where(). Values not in the mask should be set to 0.
Create a histogram of the masked image. Use the following arguments to select only non-zero pixels: min=1, max=255, bins=255.
Plot the masked image and the histogram. This has been done for you.'''
# Import SciPy's "ndimage" module
import scipy.ndimage as ndi

# Screen out non-bone pixels from "im"
mask_bone = im >=145
im_bone = np.where(mask_bone, im, 0)

# Get the histogram of bone intensities
hist = ndi.histogram(im_bone, min=1, max=255, bins=255)

# Plot masked image and histogram
fig, axes = plt.subplots(2,1)
axes[0].imshow(im_bone)
axes[1].plot(hist)
format_and_render_plot()

'''Tune a mask
Imperfect masks can be tuned through the addition and subtraction of pixels. SciPy includes several useful methods for accomplishing these ends. These include:

binary_dilation: Add pixels along edges
binary_erosion: Remove pixels along edges
binary_opening: Erode then dilate, "opening" areas near edges
binary_closing: Dilate then erode, "filling in" holes
For this exercise, create a bone mask then tune it to include additional pixels.

For the remaining exercises, we have run the following import for you:

import scipy.ndimage as ndi
Instructions
100 XP
Create a bone by selecting pixels from im that are greater than or equal to 145.
Use ndi.binary_dilation() to increase the size of mask_bone. Set the number of iterations to 5 to perform the dilation multiple times.
Use ndi.binary_closing() to fill in holes in mask_bone. Set the number of iterations to 5 to holes up to 10 pixels wide.
Plot the original and tuned masks.

'''
# Create and tune bone mask
mask_bone = im >= 145
mask_dilate = ndi.binary_dilation(mask_bone, iterations=5)
mask_closed = ndi.binary_closing(mask_bone, iterations=5)

# Plot masked images
fig, axes = plt.subplots(1,3)
axes[0].imshow(mask_bone)
axes[1].imshow(mask_dilate)
axes[2].imshow(mask_closed)
format_and_render_plot()

'''Filter convolutions
Filters are an essential tool in image processing. They allow you to transform images based on intensity values surrounding a pixel, rather than globally.

2D array convolution. By Michael Plotke [CC BY-SA 3.0  (https://creativecommons.org/licenses/by-sa/3.0)], from Wikimedia Commons

For this exercise, smooth the foot radiograph. First, specify the weights to be used. (These are called "footprints" and "kernels" as well.) Then, convolve the filter with im and plot the result.

Instructions 1/3
50 XP
1
2
3
Create a three by three array of filter weights. Set each element to 0.11 to perform mean filtering (also called "uniform filtering").'''

# Set filter weights
weights = [[0.11, 0.11, 0.11],
           [0.11, 0.11, 0.11],
           [0.11, 0.11, 0.11]]

# Convolve the image with the filter
im_filt = ndi.convolve(im, weights)

# Plot the images
fig, axes = plt.subplots(1,2)
axes[0].imshow(im)
axes[1].imshow(im_filt)
format_and_render_plot()

'''Filter functions
Convolutions rely on a set of weights, but filtering can also be done using functions such as the mean, median and maximum. Just like with convolutions, filter functions will update each pixel value based on its local neighborhood.

Consider the following lines of code:

im = np.array([[93, 36,  87], 
               [18, 49,  51],
               [45, 32,  63]])

im_filt = ____

assert im_filt[1,1] == 49
Which of the following statements should go in the blank so that the assert statement evaluates to True?

Instructions
50 XP
Possible Answers

ndi.maximum_filter(im, size=3)

ndi.uniform_filter(im, size=3)

ndi.percentile_filter(im, 60, size=3)

ndi.median_filter(im, size=3)   #--> true=== '''


"""Smoothing
Smoothing can improve the signal-to-noise ratio of your image by blurring out small variations in intensity. The Gaussian filter is excellent for this: it is a circular (or spherical) smoothing kernel that weights nearby pixels higher than distant ones.



The width of the distribution is controlled by the sigma argument, with higher values leading to larger smoothing effects.

For this exercise, test the effects of applying Gaussian filters to the foot x-ray before creating a bone mask.

Instructions
100 XP
Convolve im with Gaussian filters of size sigma=1 and sigma=3.
Plot the "bone masks" of im, im_s1, and im_s3 (i.e., where intensities are greater than or equal to 145).

"""

# Smooth "im" with Gaussian filters
im_s1 = ndi.gaussian_filter(im, sigma=1)
im_s3 = ndi.gaussian_filter(im, sigma=3)

# Draw bone masks of each image
fig, axes = plt.subplots(1,3)
axes[0].imshow(im >= 145)
axes[1].imshow(im_s1 >= 145)
axes[2].imshow(im_s3 >= 145)
format_and_render_plot()


'''Detect edges (1)
Filters can also be used as "detectors." If a part of the image fits the weighting pattern, the returned value will be very high (or very low).

In the case of edge detection, that pattern is a change in intensity along a plane. A filter detecting horizontal edges might look like this:

weights = [[+1, +1, +1],
           [ 0,  0,  0],
           [-1, -1, -1]]
For this exercise, create a vertical edge detector and see how well it performs on the hand x-ray (im).

Instructions
100 XP
Create a 3x3 array of filter weights that detects when intensity changes from the left to right. Use only the values 1, 0 and -1.
Convolve im with the edge detector.
Plot the horizontal edges with the seismic colormap. Use vmin=-150 and vmax=150 to control adjust your colormap scale.
Add a colorbar and render the results.

'''
# Set weights to detect vertical edges
weights = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]

# Convolve "im" with filter weights
edges = ndi.convolve(im, weights)

# Draw the image in color
plt.imshow(edges, cmap='seismic', vmin=-150, vmax=150)
plt.colorbar()
format_and_render_plot()

'''Detect edges (2)
Edge detection can be performed along multiple axes, then combined into a single edge value. For 2D images, the horizontal and vertical "edge maps" can be combined using the Pythagorean theorem:


One popular edge detector is the Sobel filter. The Sobel filter provides extra weight to the center pixels of the detector:

weights = [[ 1,  2,  1], 
           [ 0,  0,  0],
           [-1, -2, -1]]
For this exercise, improve upon your previous detection effort by merging the results of two Sobel-filtered images into a composite edge map.

Instructions
100 XP
Apply ndi.sobel() to im along the first and second axes.
Calculate the overall edge magnitude using the Pythagorean theorem. Use np.sqrt() and np.square().
Display the magnitude image. Use a grayscale colormap and set vmax to 75.

'''
# Apply Sobel filter along both axes
sobel_ax0 = ndi.sobel(im, axis=0)
sobel_ax1 = ndi.sobel(im, axis=1)

# Calculate edge magnitude
edges = np.sqrt(np.square(sobel_ax0) + np.square(sobel_ax1))

# Plot edge magnitude
plt.imshow(edges, cmap='gray', vmin=-150, vmax=75)
format_and_render_plot()




###################################
# 3- Measurement
###################################
'''Segment the heart
In this chapter, we'll work with magnetic resonance (MR) imaging data from the Sunnybrook Cardiac Dataset. The full image is a 3D time series spanning a single heartbeat. These data are used by radiologists to measure the ejection fraction: the proportion of blood ejected from the left ventricle during each stroke.

To begin, segment the left ventricle from a single slice of the volume (im). First, you'll filter and mask the image; then you'll label each object with ndi.label().

This chapter's exercises have the following imports:

import imageio
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
Instructions 1/2
50 XP
1
2
Apply a median filter to im. Set the size to 3.
Create a mask of values greater than 60, then use ndi.binary_closing() to fill small holes in it.
Extract a labeled array and the number of labels using ndi.label().
'''
import imageio
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
# Smooth intensity values
im_filt = ndi.median_filter(im, size=3)
# Select high-intensity pixels
mask_start = np.where(im_filt>60, 1, 0)
mask = ndi.binary_closing(mask_start)

# Label the objects in "mask"
labels, nlabels = ndi.label(mask)
print('Num. Labels:', nlabels)

'''Plot the labels array on top of the original image. To create an overlay, use np.where to convert values of 0 to np.nan. Then, plot the overlay with the rainbow colormap and set alpha=0.75 to make it transparent.'''

# Create a `labels` overlay
overlay = np.where(labels>0, labels, np.nan)

# Use imshow to plot the overlay
plt.imshow(overlay, cmap='rainbow', alpha=0.75)
format_and_render_plot()


""" #Select objects
Labels are like object "handles" - they give you a way to pick up whole sets of pixels at a time. To select a particular object:

Find the label value associated with the object.
Create a mask of matching pixels.
For this exercise, create a labeled array from the provided mask. Then, find the label value for the centrally-located left ventricle, and create a mask for it.

Instructions
100 XP
Use ndi.label() to assign labels to each separate object in mask.
Find the index value for the left ventricle label by checking the center pixel (128, 128).
Create a mask of pixels matching the left ventricle label. Using np.where, set pixels labeled as lv_val to 1 and other values to np.nan.
Use plt.imshow() to overlay the selected label on the current plot.
"""

# Label the image "mask"
labels, nlabels = ndi.label(mask)

# Select left ventricle pixels
lv_val = labels[128, 128]
lv_mask = np.where(labels == lv_val, 1, np.nan)

# Overlay selected label
plt.imshow(lv_mask, cmap='rainbow')
plt.show()

"""Extract objects
Extracting objects from the original image eliminates unrelated pixels and provides new images that can be analyzed independently.

The key is to crop images so that they only include the object of interest. The range of pixel indices that encompass the object is the bounding box.

For this exercise, use ndi.find_objects() to create a new image containing only the left ventricle.

Instructions 1/3
25 XP
1
2
3
Create the labels array from mask, then create a mask left ventricle pixels. (Use the coordinates 128, 128 to find the left ventricle label value.)"""


# Create left ventricle mask
labels, nlabels = ndi.label(mask)
lv_val = labels[128, 128]
lv_mask = np.where(labels == lv_val, 1, 0)

# Find bounding box of left ventricle
bboxes = ndi.find_objects(lv_mask)
print('Number of objects:', len(bboxes))
print('Indices for first box:', bboxes[0])

# Crop to the left ventricle (index 0)
im_lv = im[bboxes[0]]

# Plot the cropped image
plt.imshow(im_lv)
format_and_render_plot()

###
"""
Measure variance
SciPy measurement functions allow you to tailor measurements to specific sets of pixels:

Specifying labels restricts the mask to non-zero pixels.
Specifying index value(s) returns a measure for each label value.
For this exercise, calculate the intensity variance of vol with respect to different pixel sets. We have provided the 3D segmented image as labels: label 1 is the left ventricle and label 2 is a circular sample of tissue.

Labeled Volume

After printing the variances, select the true statement from the answers below.

Instructions 1/2
50 XP
1
2
Using vol and labels arrays, measure the variance of pixel intensities in the specified sets of pixels. Print them to the screen.
"""

# Variance for all pixels
var_all = ndi.variance(vol, labels=None, index=None)
print('All pixels:', var_all)

# Variance for labeled pixels
var_labels = ndi.variance(vol, labels, index=None)
print('Labeled pixels:', var_labels)

# Variance for each object
var_objects = ndi.variance(vol, labels, index=[1,2])
print('Left ventricle:', var_objects[0])
print('Other tissue:', var_objects[1])



'''
Separate histograms
A poor tissue segmentation includes multiple tissue types, leading to a wide distribution of intensity values and more variance.

On the other hand, a perfectly segmented left ventricle would contain only blood-related pixels, so the histogram of the segmented values should be roughly bell-shaped.

For this exercise, compare the intensity distributions within vol for the listed sets of pixels. Use ndi.histogram, which also accepts labels and index arguments.

Instructions 2/2
0 XP
Plot each histogram using plt.plot(). For each one, rescale by the total number of pixels to allow comparisons between them.
'''




# Create histograms for selected pixels
hist1 = ndi.histogram(vol, min=0, max=255, bins=256)
hist2 = ndi.histogram(vol, 0, 255, 256, labels=labels)
hist3 = ndi.histogram(vol, 0, 255, 256, labels=labels, index=1)

# Plot the histogram density
plt.plot(hist1 / hist1.sum(), label='All pixels')
plt.plot(hist2 / hist2.sum(), label='All labeled pixels')
plt.plot(hist3 / hist3.sum(), label='Left ventricle')
format_and_render_plot()


"""Calculate volume
Quantifying tissue morphology, or shape is one primary objective of biomedical imaging. The size, shape, and uniformity of a tissue can reveal essential health insights.

For this exercise, measure the volume of the left ventricle in one 3D image (vol).

First, count the number of voxels in the left ventricle (label value of 1). Then, multiply it by the size of each voxel in mm. (Check vol.meta for the sampling rate.)

Instructions
35 XP
Possible Answers

6,459 mm

117,329 mm

120,731 mm

18,692 mm

Submit Answer

Hint
Calculate the number of pixels in the left ventricle by calling ndi.sum(1, labels, index=1).
Calculate the unit volume for each pixel by multiplying the sampling rates from vol.meta['sampling'] together.
Calculate the volume by multiplying the number of pixels by their unit volume.
Did you find this hint helpful?


Yes

No"""



"""Calculate distance
A distance transformation calculates the distance from each pixel to a given point, usually the nearest background pixel. This allows you to determine which points in the object are more interior and which are closer to edges.

For this exercise, use the Euclidian distance transform on the left ventricle object in labels.

Instructions
100 XP
Create a mask of left ventricle pixels (Value of 1 in labels).
Calculate the distance to background for each pixel using ndi.distance_transform_edt(). Supply pixel dimensions to the sampling argument.
Print out the maximum distance and its coordinates using ndi.maximum and ndi.maximum_position.
Overlay a slice of the distance map on the original image. This has been done for you."""

# Calculate left ventricle distances
lv = np.where(labels == 1, 1, 0)
dists = ndi.distance_transform_edt(lv, sampling = vol.meta['sampling'])

# Report on distances
print('Max distance (mm):', ndi.maximum(dists))
print('Max location:', ndi.maximum_position(dists))

# Plot overlay of distances
overlay = np.where(dists[5] > 0, dists[5], np.nan)
plt.imshow(overlay, cmap='hot')
format_and_render_plot()



"""Pinpoint center of mass
The distance transformation reveals the most embedded portions of an object. On the other hand, ndi.center_of_mass() returns the coordinates for the center of an object.

The "mass" corresponds to intensity values, with higher values pulling the center closer to it.

For this exercise, calculate the center of mass for the two labeled areas. Then, plot them on top of the image.

Instructions
0 XP
Using vol and labels, calculate the center of mass for the two labeled objects. Print the coordinates.
Use plt.scatter() to add the center of mass markers to the plot. Note that scatterplots draw from the bottom-left corner. Image columns correspond to x values and rows to y values.


"""



# Extract centers of mass for objects 1 and 2
coms = ndi.center_of_mass(vol, labels, index=[1,2])
print('Label 1 center:', coms[0])
print('Label 2 center:', coms[1])

# Add marks to plot
for c0, c1, c2 in coms:
    plt.scatter(c2, c1, s=100, marker='o')
plt.show()

## time series

"""Summarize the time series
The ejection fraction is the proportion of blood squeezed out of the left ventricle each heartbeat. To calculate it, radiologists have to identify the maximum volume (systolic volume) and the minimum volume (diastolic volume) of the ventricle.

Slice 4 of Cardiac Timeseries

For this exercise, create a time series of volume calculations. There are 20 time points in both vol_ts and labels. The data is ordered by (time, plane, row, col).

Instructions
100 XP
Initialize an empty array with 20 elements using np.zeros().
Calculate the volume of each image voxel. (Consult the meta dictionary for sampling rates.)
For each time point, count the pixels in labels, and update the time series array.
Plot the time series using plt.plot().

"""

# Create an empty time series
ts = np.zeros(20)

# Calculate volume at each voxel
d0, d1, d2, d3 = vol_ts.meta['sampling']
dvoxel = d1*d2*d3

# Loop over the labeled arrays
for t in range(20):
    nvoxels = ndi.sum(1, labels[t],index=1 )
    ts[t] = nvoxels * dvoxel

# Plot the data
plt.plot(ts)
format_and_render_plot()

"""Measure ejection fraction
The ejection fraction is defined as:

 

…where  is left ventricle volume for one 3D timepoint.

To close our investigation, plot slices from the maximum and minimum volumes by analyzing the volume time series (ts). Then, calculate the ejection fraction.

After calculating the ejection fraction, review the chart below. Should this patient be concerned?



Instructions 1/3
35 XP
1
2
3
Get the index of the minimum and maximum volume images using np.argmin() and np.argmax().
Plot the extreme volumes together. Display the images along the fifth plane, e.g. (vol_ts[t, 4])."""

# Get index of max and min volumes
tmax = np.argmax(ts)
tmin = np.argmin(ts)

# Plot the largest and smallest volumes
fig, axes = plt.subplots(2,1)
axes[0].imshow(vol_ts[tmax, 4], vmax=160)
axes[1].imshow(vol_ts[tmin, 4], vmax=160)
format_and_render_plots()


# Calculate ejection fraction
ej_vol = ts.max() - ts.min()
ej_frac = ej_vol / ts.max()
print('Est. ejection volume (mm^3):', ej_vol)
print('Est. ejection fraction:', ej_frac)




########################
##4 Image Comparison
# For the final chapter, you'll need to use your brain... and hundreds of others! Drawing data from more than 400 open-access MR images, you'll learn the basics of registration, resampling, and image comparison. Then, you'll use the extracted measurements to evaluate the effect of Alzheimer's Disease on brain structure.
########################
"""Translations
In this chapter, we'll leverage data use data from the Open Access Series of Imaging Studies to compare the brains of different populations: young and old, male and female, healthy and diseased.

To start, center a single slice of a 3D brain volume (im). First, find the center point in the image array and the center of mass of the brain. Then, translate the image to the center.

This chapter's exercises have all had the following imports:

import imageio
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
Instructions 1/2
50 XP
1
2
Find the center-point of im using ndi.center_of_mass().
Calculate the distance from the image center (128, 128), along each axis.
Use ndi.shift() to shift the data."""


# Find image center of mass
com = ndi.center_of_mass(im)

# Calculate amount of shift needed
d0 = 128 - com[0]
d1 = 128 - com[1]

# Translate the brain towards the center
xfm = ndi.shift(im, shift=(d0, d1))

# Plot the original and adjusted images
fig, axes = plt.subplots(2,1)
axes[0].imshow(im)
axes[1].imshow(xfm)
format_and_render_plot()



"""Rotations
In cases where an object is angled or flipped, the image can be rotated. Using ndi.rotate(), the image is rotated from its center by the specified degrees from the right horizontal axis.

For this exercise, shift and rotate the brain image (im) so that it is roughly level and "facing" the right side of the image.
Instructions
100 XP
Shift im towards the center: 20 pixels left and 20 pixels up.
Use ndi.rotate to turn xfm 30 degrees downward. Set reshape=False to prevent the image shape from changing.
Plot the original and transformed images"""

# Shift the image towards the center
xfm = ndi.shift(im, shift=(-20, -20))

# Rotate the shifted image
xfm = ndi.rotate(xfm, angle=-30, reshape=False)

# Plot the original and rotated images
fig, axes = plt.subplots(2, 1)
axes[0].imshow(im)
axes[1].imshow(xfm)
format_and_render_plot()


"""Affine transform
An affine transformation matrix provides directions for up to four types of changes: translating, rotating, rescaling and shearing. The elements of the matrix map the coordinates from the input array to the output.

Encoded transformations within a matrix

For this exercise, use ndi.affine_transform() to apply the following registration matrices to im. Which one does the best job of centering, leveling and enlarging the original image?

Instructions
50 XP
Possible Answers

[[1, 0, 0], [0, 1, 0], [0, 0, 1]]

[[1.5, -0.8, 60], [0.8, 1.5, -140], [0, 0, 1]]

[[1, -0.3, 60], [-0.3, 1, 60], [0, 0, 1]]

--> [[0.8, -0.4, 90], [0.4, 0.8, -6.0], [0, 0, 1]]"""

"""
Resampling
Images can be collected in a variety of shapes and sizes. Resampling is a useful tool when these shapes need to be made consistent. Two common applications are:

Downsampling: combining pixel data to decrease size
Upsampling: distributing pixel data to increase size
For this exercise, transform and then resample the brain image (im) to see how it affects image shape.

Instructions
100 XP
Shift im 20 pixels left and 20 pixels up, i.e. (-20, -20). Then, rotate it 35 degrees downward. Remember to specify a value for reshape.
Use ndi.zoom() to downsample the image from (256, 256) to (64, 64).
Use ndi.zoom() to upsample the image from (256, 256) to (1024, 1024).
Plot the resampled images.

Hint
If the resampling ratio is the same in all dimensions, you can find it using: new_size / old_size. To downsample from (256, 256) to (64, 64) is 64 / 256, or 0.25.

"""


# Center and level image
xfm = ndi.shift(im, shift=(-20, -20))
xfm = ndi.rotate(xfm, angle=-35, reshape=False)

# Resample image
im_dn = ndi.zoom(xfm, zoom=0.25)
im_up = ndi.zoom(xfm, zoom=4.00)

# Plot the images
fig, axes = plt.subplots(2, 1)
axes[0].imshow(im_dn)
axes[1].imshow(im_up)
format_and_render_plot()

# You can also resample data along a single dimension by passing a tuple: e.g. ndi.zoom(im, zoom=(2,1,1)). This can be useful for making voxels cubic.

"""Interpolation
Interpolation is how new pixel intensities are estimated when an image transformation is applied. It is implemented in SciPy using sets of spline functions.

Editing the interpolation order when using a function such as ndi.zoom() modifies the resulting estimate: higher orders provide more flexible estimates but take longer to compute.

For this exercise, upsample im and investigate the effect of different interpolation orders on the resulting image.

Instructions
100 XP
Use ndi.zoom() to upsample im from a shape of 128, 128 to 512, 512 twice. First, use an interpolation order of 0, then set order to 5.
Print the array shapes of im and up0.
Plot close-ups of the images. Use the index range 128:256 along each axis."""


# Upsample "im" by a factor of 4
up0 = ndi.zoom(im, zoom=4, order=0)
up5 = ndi.zoom(im, zoom=4, order=5)

# Print original and new shape
print('Original shape:', im.shape)
print('Upsampled shape:', up5.shape)

# Plot close-ups of the new images
fig, axes = plt.subplots(1, 2)
axes[0].imshow(up0[128:256, 128:256])
axes[1].imshow(up5[128:256, 128:256])
format_and_render_plots()


### COMPARING IMAGES

"""Mean absolute error
Cost functions and objective functions output a single value that summarizes how well two images match.

The mean absolute error (MAE), for example, summarizes intensity differences between two images, with higher values indicating greater divergence.

For this exercise, calculate the mean absolute error between im1 and im2 step-by-step.

Instructions 1/3
35 XP
1
2
3
Calculate the difference between im1 and im2.
Plot err with the seismic colormap. To center the colormap at 0, set vmin=-200 and vmax=200."""

# Calculate image difference
err = im1-im2

# Plot the difference
plt.imshow(err, cmap='seismic', vmin=-200, vmax=200)
format_and_render_plot()

# Calculate absolute image difference
abs_err = np.abs(im1 - im2)

# Plot the difference
plt.imshow(abs_err, cmap='seismic', vmin=-200, vmax=200)
format_and_render_plot()

# Calculate mean absolute error
mean_abs_err = np.mean(np.abs(im1 - im2))
print('MAE:', mean_abs_err)

# <script.py> output:
#     MAE: 9.2608642578125

# Well done! The MAE metric allows for variations in weighting throughout the image, which gives areas with high pixel intensities more influence on the cost calculation than others.



Intersection of the union
Another cost function is the intersection of the union (IOU). The IOU is the number of pixels filled in both images (the intersection) out of the number of pixels filled in either image (the union).

For this exercise, determine how best to transform im1 to maximize the IOU cost function with im2. We have defined the following function for you:

def intersection_of_union(im1, im2):
    i = np.logical_and(im1, im2)
    u = np.logical_or(im1, im2)
    return i.sum() / u.sum()
Note: When using ndi.rotate(), remember to pass reshape=False, so that array shapes match.

Instructions
50 XP
Possible Answers

->> Shift (-10, -10), rotate -15 deg.

Shift (10, 10), rotate -15 deg.

Shift (10, 10), rotate +15 deg.

Shift (-10, -10), rotate +15 deg.

# Great job. Remember, the core principle is that a cost function must produce a single summary value across all elements in the image. MAE and IOU are just two of the many possible ways you might compare images.

## IDENTIFIYING POTENTIAL CONFOUNDS

"""Identifying potential confounds
Once measures have been extracted, double-check for dependencies within your data. This is especially true if any image parameters (sampling rate, field of view) might differ between subjects, or you pull multiple measures from a single image.

For the final exercises, we have combined demographic and brain volume measures into a pandas DataFrame (df).

First, you will explore the table and available variables. Then, you will check for correlations between the data.

Instructions 1/4
25 XP
1
2
3
4
Print three random rows in df using the .sample() method.
"""
# Print random sample of rows
print(df.sample(3))

# Print prevalence of Alzheimer's Disease
print(df.alzheimers.value_counts())

# Print a correlation table
print(df.corr())

#Great work! There is a high correlation - nearly 0.7 - between the brain_vol and skull_vol. We should be wary of this (and other highly correlated variables) when interpreting results.

"""Testing group differences
Let's test the hypothesis that Alzheimer's Disease is characterized by reduced brain volume.

Sample Segmentations of Alzheimer's and Typical Subject

We can perform a two-sample t-test between the brain volumes of elderly adults with and without Alzheimer's Disease. In this case, the two population samples are independent from each other because they are all separate subjects.

For this exercise, use the OASIS dataset (df) and ttest_ind to evaluate the hypothesis.

Instructions 1/4
25 XP
1
2
3
4
Import ttest_ind() from scipy.stats."""

# Import independent two-sample t-test
from scipy.stats import ttest_ind

# Select data from "alzheimers" and "typical" groups
brain_alz = df.loc[df.alzheimers == True, 'brain_vol']
brain_typ = df.loc[df.alzheimers == False, 'brain_vol']

# Perform t-test of "alz" > "typ"
results = ttest_ind(brain_alz, brain_typ)
print('t = ', results.statistic)
print('p = ', results.pvalue)

# Show boxplot of brain_vol differences
df.boxplot(column='brain_vol', by='alzheimers')
plt.show()
"""There is some evidence for decreased brain volume in individuals with Alzheimer's Disease. Since the p-value for this t-test is greater than 0.05, we would not reject the null hypothesis that states the two groups are equal."""


"""Normalizing metrics
We previously saw that there was not a significant difference between the brain volumes of elderly individuals with and without Alzheimer's Disease.

But could a correlated measure, such as "skull volume" be masking the differences?

For this exercise, calculate a new test statistic for the comparison of brain volume between groups, after adjusting for the subject's skull size.

Using results.statistic and results.pvalue as your guide, answer the question: Is there strong evidence that Alzheimer's Disease is marked by smaller brain size, relative to skull size?

Instructions 1/2
50 XP
1
2
Import ttest_ind from scipy.stats.
Divide each patient's brain_vol by their skull_vol to create a normalized measure.
Extract the adjusted brain measures from each group using df.loc.
Calculate the t-statistic and p-value using ttest_ind. Be sure to pass in brain_alz first, followed by brain_typ."""

# Import independent two-sample t-test
from scipy.stats import ttest_ind

# Adjust `brain_vol` by `skull_vol`
df['adj_brain_vol'] = df.brain_vol / df.skull_vol

# Select brain measures by group
brain_alz = df.loc[df.alzheimers == True, 'adj_brain_vol']
brain_typ = df.loc[df.alzheimers == False, 'adj_brain_vol']

# Evaluate null hypothesis
results = ttest_ind(brain_alz, brain_typ)


# Congratulations! You've worked your way through several levels of biomedical image analysis and are well-prepared for tackling new datasets and problems. For more advanced tools, I recommend checking out scikit-image, which extends the capabilities of scipy for image processing. Good luck!

#### END





