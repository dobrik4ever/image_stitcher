# Microscopic Stitcher

Stitcher class allows you to stitch multiple frames of microscopic scanner together.

## Usage

Download repo, and take a look at file structure. In folder ```images``` you can put all of your images from scanner. Images must be named in manner: ```img_000x_000y.jpg``` where ```000x``` might be 0025 0001 0765 or even 2578

## Basic example

Stitcher has 3 main attributes:

1. overlap - length in pixels, denotes the overlap between two images
2. path - actually path to images folder
3. inverseY - if ```True``` inverts the image order along Y axis

```python
from Stitch import Stitcher
from matplotlib import pyplot as plt

stitcher = Stitcher(    # Initialize a new stitcher
        overlap = 200,  # Overlap of 2 neighbor images
        path = 'images',# Path to images folder
        inverseY = True)# Inverse image order along Y axis
stitcher.run()          # Run position finding process
pan = stitcher.stitch() # Stitch images together

plt.imshow(cv2.cvtColor(pan, cv2.COLOR_BGR2RGB))
plt.show()
```