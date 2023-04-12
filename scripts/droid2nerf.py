#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
from glob import glob
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SCRIPTS_FOLDER = os.path.join(ROOT_DIR, "scripts")

def parse_args():
	parser = argparse.ArgumentParser(description="Convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place.")

	parser.add_argument("--video_in", default="", help="Run ffmpeg first to convert a provided video file into a set of images. Uses the video_fps parameter also.")
	parser.add_argument("--video_fps", default=2)
	parser.add_argument("--time_slice", default="", help="Time (in seconds) in the format t1,t2 within which the images should be generated from the video. E.g.: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video.")
	parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
	parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.")
	parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
	parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")
	parser.add_argument("--colmap_camera_params", default="", help="Intrinsic parameters, depending on the chosen model. Format: fx,fy,cx,cy,dist")
	parser.add_argument("--images", default="images", help="Input path to the images.")
	parser.add_argument("--text", default="colmap_text", help="Input path to the colmap text files (set automatically if --run_colmap is used).")
	parser.add_argument("--aabb_scale", default=32, choices=["1", "2", "4", "8", "16", "32", "64", "128"], help="Large scene scale factor. 1=scene fits in unit cube; power of 2 up to 128")
	parser.add_argument("--skip_early", default=0, help="Skip this many images from the start.")
	parser.add_argument("--keep_colmap_coords", action="store_true", help="Keep transforms.json in COLMAP's original frame of reference (this will avoid reorienting and repositioning the scene for preview and rendering).")
	parser.add_argument("--out", default="./", help="Output directory.")
	parser.add_argument("--vocab_path", default="", help="Vocabulary tree path.")
	parser.add_argument("--overwrite", action="store_true", help="Do not ask for confirmation for overwriting existing images and COLMAP data.")
	parser.add_argument("--mask_categories", nargs="*", type=str, default=[], help="Object categories that should be masked out from the training images. See `scripts/category2id.json` for supported categories.")
	parser.add_argument("--droid_root", default="", required=True, help="Directory containing droid-SLAM reconstruction")
	parser.add_argument("--calib", default="", required=True, help=".txt file containing camera calibration")
	args = parser.parse_args()
	return args

def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)

def run_ffmpeg(args):
	ffmpeg_binary = "ffmpeg"

	# On Windows, if FFmpeg isn't found, try automatically downloading it from the internet
	if os.name == "nt" and os.system(f"where {ffmpeg_binary} >nul 2>nul") != 0:
		ffmpeg_glob = os.path.join(ROOT_DIR, "external", "ffmpeg", "*", "bin", "ffmpeg.exe")
		candidates = glob(ffmpeg_glob)
		if not candidates:
			print("FFmpeg not found. Attempting to download FFmpeg from the internet.")
			do_system(os.path.join(SCRIPTS_FOLDER, "download_ffmpeg.bat"))
			candidates = glob(ffmpeg_glob)

		if candidates:
			ffmpeg_binary = candidates[0]

	if not os.path.isabs(args.images):
		args.images = os.path.join(os.path.dirname(args.video_in), args.images)

	images = "\"" + args.images + "\""
	video =  "\"" + args.video_in + "\""
	fps = float(args.video_fps) or 1.0
	print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
	if not args.overwrite and (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
		sys.exit(1)
	try:
		# Passing Images' Path Without Double Quotes
		shutil.rmtree(args.images)
	except:
		pass
	do_system(f"mkdir {images}")

	time_slice_value = ""
	time_slice = args.time_slice
	if time_slice:
		start, end = time_slice.split(",")
		time_slice_value = f",select='between(t\,{start}\,{end})'"
	do_system(f"{ffmpeg_binary} -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" {images}/%04d.jpg")

def run_colmap(args):
	colmap_binary = "colmap"

	# On Windows, if FFmpeg isn't found, try automatically downloading it from the internet
	if os.name == "nt" and os.system(f"where {colmap_binary} >nul 2>nul") != 0:
		colmap_glob = os.path.join(ROOT_DIR, "external", "colmap", "*", "COLMAP.bat")
		candidates = glob(colmap_glob)
		if not candidates:
			print("COLMAP not found. Attempting to download COLMAP from the internet.")
			do_system(os.path.join(SCRIPTS_FOLDER, "download_colmap.bat"))
			candidates = glob(colmap_glob)

		if candidates:
			colmap_binary = candidates[0]

	db = args.colmap_db
	images = "\"" + args.images + "\""
	db_noext=str(Path(db).with_suffix(""))

	if args.text=="text":
		args.text=db_noext+"_text"
	text=args.text
	sparse=db_noext+"_sparse"
	print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
	if not args.overwrite and (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
		sys.exit(1)
	if os.path.exists(db):
		os.remove(db)
	do_system(f"{colmap_binary} feature_extractor --ImageReader.camera_model {args.colmap_camera_model} --ImageReader.camera_params \"{args.colmap_camera_params}\" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}")
	match_cmd = f"{colmap_binary} {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {db}"
	if args.vocab_path:
		match_cmd += f" --VocabTreeMatching.vocab_tree_path {args.vocab_path}"
	do_system(match_cmd)
	try:
		shutil.rmtree(sparse)
	except:
		pass
	do_system(f"mkdir {sparse}")
	do_system(f"{colmap_binary} mapper --database_path {db} --image_path {images} --output_path {sparse}")
	do_system(f"{colmap_binary} bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
	try:
		shutil.rmtree(text)
	except:
		pass
	do_system(f"mkdir {text}")
	do_system(f"{colmap_binary} model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

if __name__ == "__main__":
	args = parse_args()
	AABB_SCALE = int(args.aabb_scale)
	SKIP_EARLY = int(args.skip_early)

	poses_arr = np.load(os.path.join(args.droid_root, 'poses_mtx.npy'))

	OUT_PATH = os.path.join(args.out, 'transforms.json')
	print(f"outputting to {OUT_PATH}...")
	with open(args.calib, 'r') as f:
		angle_x = math.pi / 2
		for line in f:
			els = line.split(" ")
			w = 2 * float(els[2])
			h = 2 * float(els[3])
			fl_x = float(els[0])
			fl_y = float(els[1])
			k1 = float(els[4])
			k2 = float(els[5])
			k3 = 0
			k4 = 0
			p1 = float(els[6])
			p2 = float(els[7])
			cx = w / 2
			cy = h / 2

			angle_x = math.atan(w / (fl_x * 2)) * 2
			angle_y = math.atan(h / (fl_y * 2)) * 2
			fovx = angle_x * 180 / math.pi
			fovy = angle_y * 180 / math.pi

	print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

	img_list = [l.strip() for l in open(os.path.join(args.droid_root,"images.txt"), "r").readlines()]
	
	bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
	out = {
		"camera_angle_x": angle_x,
		"camera_angle_y": angle_y,
		"fl_x": fl_x,
		"fl_y": fl_y,
		"k1": k1,
		"k2": k2,
		"k3": k3,
		"k4": k4,
		"p1": p1,
		"p2": p2,
		"is_fisheye": False,
		"cx": cx,
		"cy": cy,
		"w": w,
		"h": h,
		"aabb_scale": AABB_SCALE,
		"frames": [],
	}

	up = np.zeros(3)
	for idx, line in enumerate(img_list):
		line = line.strip()
		c2w = poses_arr[idx]
		# c2w = np.linalg.inv(m)
		up += c2w[0:3,1]
		if not args.keep_colmap_coords:
			c2w[0:3,2] *= -1 # flip the y and z axis
			c2w[0:3,1] *= -1
			c2w = c2w[[1,0,2,3],:]
			c2w[2,:] *= -1 # flip whole world upside down

			up += c2w[0:3,1]

		frame = {"file_path":line, "transform_matrix": c2w}
		out["frames"].append(frame)
	nframes = len(out["frames"])

	if args.keep_colmap_coords:
		flip_mat = np.array([
			[1, 0, 0, 0],
			[0, -1, 0, 0],
			[0, 0, -1, 0],
			[0, 0, 0, 1]
		])

		for f in out["frames"]:
			f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
	else:
		# don't keep colmap coords - reorient the scene to be easier to work with
		
		up = up / np.linalg.norm(up)
		print("up vector was", up)
		R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
		R = np.pad(R,[0,1])
		R[-1, -1] = 1

		for f in out["frames"]:
			f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

		# find a central point they are all looking at
		print("computing center of attention...")
		totw = 0.0
		totp = np.array([0.0, 0.0, 0.0])
		for f in out["frames"]:
			mf = f["transform_matrix"][0:3,:]
			for g in out["frames"]:
				mg = g["transform_matrix"][0:3,:]
				p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
				if w > 0.00001:
					totp += p*w
					totw += w
		if totw > 0.0:
			totp /= totw
		print(totp) # the cameras are looking at totp
		for f in out["frames"]:
			f["transform_matrix"][0:3,3] -= totp

		avglen = 0.
		for f in out["frames"]:
			avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
		avglen /= nframes
		print("avg camera distance from origin", avglen)
		for f in out["frames"]:
			f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

	for f in out["frames"]:
		f["transform_matrix"] = f["transform_matrix"].tolist()
	print(nframes,"frames")
	print(f"writing {OUT_PATH}")
	with open(OUT_PATH, "w") as outfile:
		json.dump(out, outfile, indent=2)
