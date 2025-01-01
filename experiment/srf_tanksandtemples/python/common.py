import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve1d

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 10000000000

import flip
import flip.utils

import torch
# pip install lpips
# more details refer to `https://github.com/richzhang/PerceptualSimilarity`
import lpips
lpips_fn = None

#####################################################################################
# Color space
#####################################################################################
def srgb_to_linear(img):
	limit = 0.04045
	return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

def linear_to_srgb(img):
	limit = 0.0031308
	return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

def luminance(a):
	a = np.maximum(0, a)**0.4545454545
	return 0.2126 * a[:,:,0] + 0.7152 * a[:,:,1] + 0.0722 * a[:,:,2]

#####################################################################################
# Image I/O
#####################################################################################
def read_image_pillow(img_file):
	img = PIL.Image.open(img_file, "r")
	if os.path.splitext(img_file)[1] == ".jpg":
		img = img.convert("RGB")
	else:
		img = img.convert("RGBA")
	img = np.asarray(img).astype(np.float32)
	return img / 255.0

def write_image_pillow(img_file, img, quality):
	img_array = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
	im = PIL.Image.fromarray(img_array)
	if os.path.splitext(img_file)[1] == ".jpg":
		im = im.convert("RGB") # Bake the alpha channel
	im.save(img_file, quality=quality, subsampling=0)

def read_image(file):
	if os.path.splitext(file)[1] == ".exr":
		import pyexr as exr
		img = exr.read(file).astype(np.float32)
	elif os.path.splitext(file)[1] == ".bin":
		with open(file, "rb") as f:
			bytes = f.read()
			h, w = struct.unpack("ii", bytes[:8])
			img = np.frombuffer(bytes, dtype=np.float16, count=h*w*4, offset=8).astype(np.float32).reshape([h, w, 4])
	else:
		img = read_image_pillow(file)
		if img.shape[2] == 4:
			img[...,0:3] = srgb_to_linear(img[...,0:3])
			# Premultiply alpha
			img[...,0:3] *= img[...,3:4]
		else:
			img = srgb_to_linear(img)
	return img

def write_image(file, img, quality=95):
	if os.path.splitext(file)[1] == ".exr":
		import pyexr as exr
		img = exr.write(file, img)
	elif os.path.splitext(file)[1] == ".bin":
		if img.shape[2] < 4:
			img = np.dstack((img, np.ones([img.shape[0], img.shape[1], 4 - img.shape[2]])))
		with open(file, "wb") as f:
			f.write(struct.pack("ii", img.shape[0], img.shape[1]))
			f.write(img.astype(np.float16).tobytes())
	else:
		if img.shape[2] == 4:
			img = np.copy(img)
			# Unmultiply alpha
			img[...,0:3] = np.divide(img[...,0:3], img[...,3:4], out=np.zeros_like(img[...,0:3]), where=img[...,3:4] != 0)
			img[...,0:3] = linear_to_srgb(img[...,0:3])
		else:
			img = linear_to_srgb(img)
		write_image_pillow(file, img, quality)

def write_image_gamma(file, img, gamma, quality=95):
	if os.path.splitext(file)[1] == ".exr":
		import pyexr as exr
		img = exr.write(file, img)
	else:
		img = img**(1.0/gamma) # this will break alpha channels
		write_image_pillow(file, img, quality)

def write_depth(file, depth, scale=1.0, cm=None):
	depth = np.nan_to_num(depth).astype(np.float32)
	depth *= scale
	if cm is not None:
		depth_colorized = plt.get_cmap(cm)(depth)
		write_image_pillow(file, depth_colorized, 95)
	else:
		with open(file, 'wb') as f:
			PIL.Image.fromarray(depth).save(f, 'TIFF')

#####################################################################################
# Image Metrics
#####################################################################################
def L1(img, ref):
	return np.abs(img - ref)

def APE(img, ref):
	return L1(img, ref) / (1e-2 + ref)

def SAPE(img, ref):
	return L1(img, ref) / (1e-2 + (ref + img) / 2.)

def L2(img, ref):
	return (img - ref)**2

def RSE(img, ref):
	return L2(img, ref) / (1e-2 + ref**2)

def trim(error, skip=0.000001):
	error = np.sort(error.flatten())
	size = error.size
	skip = int(skip * size)
	return error[skip:size-skip].mean()

def SSIM(a, b):
	def blur(a):
		k = np.array([0.120078, 0.233881, 0.292082, 0.233881, 0.120078])
		x = convolve1d(a, k, axis=0)
		return convolve1d(x, k, axis=1)
	a = luminance(a)
	b = luminance(b)
	mA = blur(a)
	mB = blur(b)
	sA = blur(a*a) - mA**2
	sB = blur(b*b) - mB**2
	sAB = blur(a*b) - mA*mB
	c1 = 0.01**2
	c2 = 0.03**2
	p1 = (2.0*mA*mB + c1)/(mA*mA + mB*mB + c1)
	p2 = (2.0*sAB + c2)/(sA + sB + c2)
	error = p1 * p2
	return error

def compute_error_img(metric, img, ref):
	img[np.logical_not(np.isfinite(img))] = 0
	img = np.maximum(img, 0.)
	if metric == "MAE":
		return L1(img, ref)
	elif metric == "MAPE":
		return APE(img, ref)
	elif metric == "SMAPE":
		return SAPE(img, ref)
	elif metric == "MSE":
		return L2(img, ref)
	elif metric == "MScE":
		return L2(np.clip(img, 0.0, 1.0), np.clip(ref, 0.0, 1.0))
	elif metric == "MRSE":
		return RSE(img, ref)
	elif metric == "MtRSE":
		return trim(RSE(img, ref))
	elif metric == "MRScE":
		return RSE(np.clip(img, 0, 100), np.clip(ref, 0, 100))
	elif metric == "SSIM":
		return SSIM(np.clip(img, 0.0, 1.0), np.clip(ref, 0.0, 1.0))
	elif metric in ["FLIP", "\FLIP"]:
		# Set viewing conditions
		monitor_distance = 0.7
		monitor_width = 0.7
		monitor_resolution_x = 3840
		# Compute number of pixels per degree of visual angle
		pixels_per_degree = monitor_distance * (monitor_resolution_x / monitor_width) * (np.pi / 180)

		ref_srgb = np.clip(flip.color_space_transform(ref, "linrgb2srgb"), 0, 1)
		img_srgb = np.clip(flip.color_space_transform(img, "linrgb2srgb"), 0, 1)
		result = flip.compute_flip(flip.utils.HWCtoCHW(ref_srgb), flip.utils.HWCtoCHW(img_srgb), pixels_per_degree)
		assert np.isfinite(result).all()
		return flip.utils.CHWtoHWC(result)

	raise ValueError(f"Unknown metric: {metric}.")

def mse2psnr(x): return -10.*np.log(x)/np.log(10.)

def compute_lpips(img, ref):
	assert img.dtype==np.float32 and ref.dtype==np.float32
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

	# normalized to [-1,1]
	img, ref = img*2.-1., ref*2.-1.

	global lpips_fn
	if lpips_fn is None: lpips_fn = lpips.LPIPS(net='vgg').to(device)

	error = lpips_fn.forward(
		torch.Tensor((img)[None].transpose(0,3,1,2)).to(device),
		torch.Tensor((ref)[None].transpose(0,3,1,2)).to(device))
	return error.cpu().item()

def compute_error(metric, img, ref):
	if metric == 'LPIPS':
		error = compute_lpips(img, ref)
		return error, None
	metric_map = compute_error_img(metric, img, ref)
	metric_map[np.logical_not(np.isfinite(metric_map))] = 0
	if len(metric_map.shape) == 3:
		metric_map = np.mean(metric_map, axis=2)
	mean = np.mean(metric_map)
	return mean, metric_map