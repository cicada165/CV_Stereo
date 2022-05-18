import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo (or mean albedo for RGB input images) for a pixel is less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights. All N images have the same dimension.
                  The images can either be all RGB (height x width x 3), or all grayscale (height x width x 1).

    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    imageShape = np.shape(images)
    I = np.array(images).reshape(imageShape[0], np.prod(imageShape[1:]))
    Lt_I = np.dot(lights.T, I)
    Lt_L = np.dot(lights.T, lights)

    #Compute G: G = Lt_L_Inv * Lt_I
    G = np.dot(np.linalg.inv(Lt_L), Lt_I)

    #Albedo
    tmp = [G.shape[0]]
    tmp.extend(imageShape[1:])
    G_albedo = G.reshape(tmp)

    albedo = np.linalg.norm(G_albedo, axis = 0)

    #Normal
    tmp = []
    tmp.extend(imageShape[1:])
    tmp.append(3)

    G_normal = np.mean(G.T.reshape(tmp),axis = 2)
    albedo_normal = np.linalg.norm(G_normal, axis = 2)
    normal = G_normal/(albedo_normal[:, :, None]).astype(float)
    
    normal = np.nan_to_num(normal)
  
    return albedo, normal
    # raise NotImplementedError()


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """


    height = points.shape[0]
    width = points.shape[1]
    projections = np.zeros((height, width, 2))

    k_dot_rt = K.dot(Rt)
    for i in range(height):
        for j in range(width):
            point = points[i,j]
            p = np.array([point[0], point[1], point[2], 1.0])

            homogenous_cor = k_dot_rt.dot(p)
            h1 = homogenous_cor[0]
            h2 = homogenous_cor[1]
            h3 = homogenous_cor[2]

            projections[i,j] = np.array([h1/h3, h2/h3])

    return projections

    #raise NotImplementedError()


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc_impl' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc_impl' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """


    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    normalized = np.zeros((height, width, (channels * ncc_size**2)))

    bound = ncc_size//2

    leftboundH = bound
    rightboundH = height - bound

    leftboundW = bound
    rightboundW = width - bound
    

    for i in range(leftboundH, rightboundH):
        for j in range(leftboundW, rightboundW):  
            index = 0
            patch = image[i - bound: i+bound + 1, j - bound: j+bound+1, :]
            patch = patch - np.mean(np.mean(patch, axis=0), axis=0)
            
            v = np.zeros((channels * ncc_size**2))
        
            for channel in range(channels):
                for pi in range(ncc_size):
                    for pj in range(ncc_size):
                        v[index] = patch[pi, pj, channel]
                        index = index + 1
                
            norm = np.linalg.norm(v)

            if(norm >= 1e-6):
                v = v / norm
                normalized[i,j] = v

    return normalized



    #raise NotImplementedError()


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc_impl.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """

    ncc = np.sum(np.multiply(image1,image2),axis = 2)
    return ncc
    # raise NotImplementedError()
