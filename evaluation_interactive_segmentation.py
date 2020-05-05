

# Evaluation using the F-measure 


import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os
import statistics
import pickle



src_folder = '/media/minh/DATA/Study/Results/segmentation_scribble/Segment_2020/Dahu_MRF/lambda05/'
gt_folder  = '/media/minh/DATA/Study/database/Interative_Dataset/images-gt/images-gt/'
label_folder  = '/media/minh/DATA/Study/database/Interative_Dataset/images-labels/images-labels/'
# filled_folder = '/media/minh/DATA/Study/Results/segmentation_scribble/Segment_2020/filled_post_fgbg_logpro/'



input_file = os.listdir(src_folder)

print(len(input_file))

F_measure = []

result = []

for entry in input_file:
	

	print(entry)

	parts = entry.split(".")
	

	src_name = src_folder + entry
	gt_name  = gt_folder  + parts[0] + '.png'
	label_name  = label_folder  + parts[0] + '-anno.png'

	#print(src_name)
	#print(gt_name)

	src_image = cv2.imread(src_name)
	gt_image = cv2.imread(gt_name)	
	label = cv2.imread(label_name)
	#print(src_image.shape)


	# Read Image
	srcImg = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
	gtImg  = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
	label_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
	label_gray[label_gray>100] = 255

	w,h = gtImg.shape

	
	#print(w,h)
	# resize image size

	srcImg = cv2.resize(srcImg, (h, w) )


	# threshold image

	ret, srcImg = cv2.threshold(srcImg, 20, 255, 0)
	ret, gtImg = cv2.threshold(gtImg, 20, 255, 0)




	# Find largest contour in intermediate image so that it contains the markers 
	cnts, _ = cv2.findContours(srcImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	con = []
	for i in range(len(cnts)):
		biggest = np.zeros(srcImg.shape, np.uint8)
		cv2.drawContours(biggest, [cnts[i]], -1, 255, cv2.FILLED)
		biggest = cv2.bitwise_and(label_gray, biggest)
		if (np.sum(biggest) > 0):
			con.append(cnts[i])

	print(len(con))

	if (len(con) == 0):
		F = 0
	else:

		cnt = max(con, key=cv2.contourArea)

		# Output
		biggest = np.zeros(srcImg.shape, np.uint8)
		cv2.drawContours(biggest, [cnt], -1, 255, cv2.FILLED)
		#biggest = cv2.bitwise_and(srcImg, biggest)

		# score1_name = filled_folder + entry
		# cv2.imwrite(score1_name, biggest)	

		biggest = biggest/255
		gtImg = gtImg/255

		# Compute F-measure


		beta = 1
		sum_src = np.sum(biggest)
		sum_gt  = np.sum(gtImg)
		intersection = cv2.bitwise_and(gtImg, biggest)
		sum_inter = np.sum(intersection)

		precision = float(sum_inter)/sum_src
		recall    = float(sum_inter)/sum_gt
		F = float((1+ beta)*precision * recall)  /(beta *precision +recall+0.000001)
		
	F_measure.append(F) 

	print(F)

	result.append([entry, F])



F_measure_average = sum(F_measure) / len(F_measure)

std_mesure = statistics.stdev(F_measure)

print(F_measure_average, std_mesure)


# import csv


# with open("Grabcut.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(result)


# plt.imshow(intersection, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()







