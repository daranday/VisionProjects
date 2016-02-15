import numpy as np
import matplotlib.pyplot as plt
import imutils
import cv2

class Stitcher(object):
	"""Stitcher class forms a panorama from images"""
	def __init__(self):
		pass

	def stitch(self, images, showMatches = True):
		(imageA, imageB) = images
		
		(kpsA, ftsA) = self.detectAndDescribe(imageA)
		(kpsB, ftsB) = self.detectAndDescribe(imageB)

		M = self.matchKeypoints(kpsA, kpsB, ftsA, ftsB)

		if M == None:
			return None

		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		result = cv2.warpPerspective(imageA, H,
			(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
 
		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)
 
			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)
 
		# return the stitched image
		return result

	def detectAndDescribe(self, image):
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		
		# detect keypoints in the image
		detector = cv2.FeatureDetector_create("SIFT")
		kps = detector.detect(gray)

		# extract features from the image
		extractor = cv2.DescriptorExtractor_create("SIFT")
		(kps, features) = extractor.compute(gray, kps)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])
 		
 		# return a tuple of keypoints and features
		return (kps, features)
		
	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio = 0.75, reprojThresh = 4.0):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
 
			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
 
			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)
 
		# otherwise, no homograpy could be computed
		return None
		
	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
 
		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
 
		# return the visualization
		return vis

if __name__ == "__main__":
	stitcher = Stitcher()

	images = [cv2.imread('shot1.jpg'), cv2.imread('shot2.jpg')]
	images = [imutils.resize(image, width=1000) for image in images]
	images.reverse()

	stitcher = Stitcher()
	(result, vis) = stitcher.stitch(images, showMatches=True)

	# show the images
	# cv2.imshow("Image A", images[0])
	# cv2.imshow("Image B", images[1])
	cv2.imshow("Keypoint Matches", vis)
	cv2.imshow("Result", result)
	cv2.waitKey(0)
 


# gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray', gray1)
# if cv2.waitKey(0) & 0xff == 27:
# 	cv2.destroyAllWindows()


# dst = cv2.cornerHarris(gray1,2,3,0.04)
# dst = cv2.dilate(dst,None)
# img1[dst>0.01*dst.max()]=[0,0,255]

# # cv2.imshow('dst',img1)

# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray1,None)

# img=cv2.drawKeypoints(gray,kp)

# cv2.imwrite('sift_keypoints.jpg',img)

# if cv2.waitKey(0) & 0xff == 27:
# 	cv2.destroyAllWindows()

# # plt.imshow(img1)
# # plt.show()
