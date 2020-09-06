import cv2
from numpy import zeros, ones, float32, uint8, linalg, dot
from matplotlib.pyplot import *
import os


class Image:

    def __init__(self, fname, iy, ix, y=0, x=0):
        self.fname = fname
        self.iy = int(iy)
        self.ix = int(ix)
        self.y = y
        self.x = x
        self.pos = (y, x)

    def __str__(self):
        s = f'File name: {self.fname}, pos: {self.pos}'
        return s


class Stitcher:

    def __init__(self, overlap: int, path: str, inverseY: bool = True):
        '''
        Images must be named_like:
        img_0_0.jpg
        img_59_62.jpg
        '''
        self.draw_matches = False
        self.overlap = overlap
        self.path = path
        self.image_matches = []
        self.MAX_FEATURES = 8000
        self.GOOD_MATCH_PERCENT = 0.005
        self.images = []
        self.inverseY = inverseY
        self.__file_finder()

    def __file_finder(self):
        H, W = 0, 0
        mx, my = 10e9, 10e9
        files = [file for file in os.listdir(self.path) if file.endswith(
            '.jpg') and file.startswith('img_')]
        for file in files:
            _, x, y = file.split('.')[0].split('_')
            H = max(H, int(y))
            W = max(W, int(x))
            mx = min(mx, int(x))
            my = min(my, int(y))
        for file in files:
            _, x, y = file.split('.')[0].split('_')
            iy = int(y) + 1
            if self.inverseY:
                iy = H - int(y) + my + 1
            iy = int(iy) - my
            x = int(x) - mx + 1
            nfile = f'imgi_{x:04}_{iy:04}.jpg'
            os.rename(self.path+'/'+file, self.path+'/'+nfile)

        files = [file for file in os.listdir(
            self.path) if file.endswith('.jpg')]
        
        self.H, self.W = 0, 0
        for file in files:
            _, x, y = file.split('.')[0].split('_')
            img = Image(
                fname=self.path+'/'+file,
                iy=y,
                ix=x)
            self.images.append(img)
            self.H = max(self.H, int(y))
            self.W = max(self.W, int(x))
        self.imageHeight, self.imageWidth, _ = cv2.imread(
            self.images[-1].fname).shape

    def get_overlap_areas(self, img1, img2, direction):
        im1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if direction == 'horizontal':
            im1Gray = im1Gray[:, -self.overlap:]
            im2Gray = im2Gray[:, :self.overlap]
        if direction == 'vertical':
            im1Gray = im1Gray[-self.overlap:, :]
            im2Gray = im2Gray[:self.overlap, :]

        return im1Gray, im2Gray

    def get_keypoints(self, im1Gray, im2Gray):

        orb = cv2.ORB_create(self.MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
        matches = matcher.match(descriptors1, descriptors2, None)
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        points1 = zeros((len(matches), 2), dtype=float32)
        points2 = zeros((len(matches), 2), dtype=float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
        if self.draw_matches:
            imMatches = cv2.drawMatches(
                im1Gray, keypoints1, im2Gray, keypoints2, matches, None)
            self.image_matches.append(imMatches)
            figure(figsize=(8, 8))
            imshow(imMatches)
            show()

        return points1, points2

    def get_translation(self, p1, p2):
        p11 = ones([p1.shape[0], 3]).astype(float32)
        p21 = ones([p2.shape[0], 3]).astype(float32)

        p11[:, :-1] = p1
        p21[:, :-1] = p2

        p2p = linalg.pinv(p21)
        T = dot(p2p, p11)
        T = T.T
        tx = int(T[0, 2])
        ty = int(T[1, 2])

        return tx, ty

    def add_image(self, img, y, x):
        h, w, _ = img.shape
        try:
            self.pan[y:y+h, x:x+w, :] = img
        except Exception as e:
            print(e)
            print('OOps...', 'y,x', y, x)

    def get_new_coords(self, origin, shift, direction):
        y1, x1 = origin
        ty, tx = shift
        if direction == 'horizontal':
            x2 = x1 + self.imageWidth + tx - self.overlap
            y2 = y1 + ty
        elif direction == 'vertical':
            x2 = x1 + tx
            y2 = y1 + self.imageHeight + ty - self.overlap
        else:
            raise ValueError('Invalid direction')
        return y2, x2

    def run(self, draw_matches = False):
        self.draw_matches = draw_matches

        def get_coord(direction, prevImg, img, origin):
            ig1, ig2 = self.get_overlap_areas(prevImg, img, direction)
            p1, p2 = self.get_keypoints(ig1, ig2)
            tx, ty = self.get_translation(p1, p2)
            origin = self.get_new_coords(origin, (ty, tx), direction)
            return origin

        origin = (0, 0)

        for ix in range(self.W):
            for iy in range(self.H):
                itr = iy + self.H * ix
                img = cv2.imread(self.images[itr].fname)
                if ix == 0 and iy == 0:
                    direction = 'vertical'
                    prevImg = img
                    self.images[itr].pos = (0, 0)
                    continue
                elif iy == 0:
                    prevImage = self.images[itr-self.H]
                    prevImg = cv2.imread(prevImage.fname)
                    origin = prevImage.pos
                    direction = 'horizontal'
                else:
                    direction = 'vertical'

                origin = get_coord(direction, prevImg, img, origin)
                prevImg = img
                self.images[itr].pos = origin

        return self.images

    def stitch(self):
        minx, miny = 0, 0
        maxx, maxy = 0, 0
        for i in self.images:
            y, x = i.pos
            minx = min(x, minx)
            maxx = max(x, maxx)
            miny = min(y, miny)
            maxy = max(y, maxy)
        maxx -= minx
        maxy -= miny
        height = maxy + self.imageHeight
        width = maxx + self.imageWidth
        self.pan = zeros([height, width, 3]).astype(uint8)
        print(self.pan.shape)
        for i in self.images:
            y, x = i.pos
            x -= minx
            y -= miny
            img = cv2.imread(i.fname)
            self.add_image(img, y, x)
        return self.pan

if __name__ == '__main__':
    S = Stitcher(
        overlap = 200,
        path = 'images',
        inverseY = True)
    S.run()
    pan = S.stitch()
    imshow(cv2.cvtColor(pan, cv2.COLOR_BGR2RGB))
    show()
