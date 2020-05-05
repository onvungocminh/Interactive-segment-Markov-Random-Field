import sys
import cv2
import numpy as np
import maxflow
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
from dijkstar import Graph, find_path
import os
import preprocess_minh
import dahu


import matplotlib.pyplot as plt




def eu_dis(v1, v2):
    return np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2)



def computeEnerge(label, fg, bg1,  neighbor, lamda,  LAB_map, sigma1):
    return (giveDataEnerge(label, fg, bg1) + giveSmoothEnerge(label, neighbor, lamda,  LAB_map, sigma1))

def giveDataEnerge( label, fg, bg1):
    energe = 0
    h,w = fg.shape

    for x in range (h):
        for y in range (w):
            if label[x][y] == 1:
                energe += fg[x][y]
            elif label[x][y] == 0:
                energe += bg1[x][y]

    return energe


def giveSmoothEnerge(label, neighbor, lamda,  LAB_map, sigma1 ):  # compute SmoothEnerge
    energe = 0
    h,w = label.shape
    for x in range (h):
        for y in range (w):
            u = x*w + y
            for i in range (4):
                a = x + neighbor[i][0]
                b = y + neighbor[i][1]
                if (a >= 0 and a <h and b >= 0 and b < w):
                    v = a*w + b
                    if v < u:
                        continue
                    if label[x][y] == label[a][b]:
                        continue
                    energe += lamda * np.e ** (-(eu_dis(LAB_map[x][y], LAB_map[a][b]) ** 2) / sigma1)
    return energe







src_folder     = '/media/minh/DATA/Study/database/Interative_Dataset/images/images/'
label_folder   = '/media/minh/DATA/Study/database/Interative_Dataset/images-labels/images-labels/'
output_folder  = '/media/minh/DATA/Study/Results/segmentation_scribble/Segment_2020/Dahu_MRF/lambda05/'
output_post_folder  = '/media/minh/DATA/Study/Results/segmentation_scribble/Segment_2020/Dahu_MRF/post/'


input_file = os.listdir(src_folder)

print(len(input_file))

for entry in input_file:
    
    print(entry)



    parts = entry.split(".")
    

    src_name = src_folder + entry
    label_name  = label_folder  + parts[0] + '-anno.png'





    img = cv2.imread(src_name)
    label_gray = cv2.cvtColor(cv2.imread(label_name), cv2.COLOR_BGR2GRAY)
    ima_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)


    h, w = label_gray.shape
    print(h, w)





    ###### LAB map

    LAB_map_raw = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    LAB_map = np.zeros_like(LAB_map_raw, dtype=np.int8)
    for i in range(len(LAB_map)):
        for j in range(len(LAB_map[0])):
            LAB_map[i, j][0] = LAB_map_raw[i, j][0] / 255 * 100
            LAB_map[i, j][1] = LAB_map_raw[i, j][1] - 128
            LAB_map[i, j][2] = LAB_map_raw[i, j][2] - 128



    ##### confidence map



    list_bg= []
    list_fg= []

    bg_markers = []
    fg_markers = []

    for i in range (0, h):
        for j in range (0, w):
            if (label_gray[i][j] > 10 and label_gray[i][j] < 100 ):
                list_bg.append(ima_lab[i][j])
                bg_markers.append([j,i])
                
            if (label_gray[i][j] > 100):
                list_fg.append(ima_lab[i][j])
                fg_markers.append([j,i])            

    list_bg = np.asarray(list_bg)
    list_fg = np.asarray(list_fg)

    bg_markers = np.array(bg_markers)
    fg_markers = np.array(fg_markers)



    score, score1 = preprocess_minh.preprocess_logpro(ima_lab, list_fg, list_bg)
    print("log of probability")




    # background

    # score = np.pad(score, 1, 'constant', constant_values=m0)
    score = np.array(score * 255, dtype="uint8") # convert to uint8
    #score = cv2.bilateralFilter(score,9,75,75)
    app = dahu.DahuApplication(score)
    app.setMarkers(fg_markers, bg_markers)
    fg = np.array(app.fg, dtype="float32")
    bg = np.array(app.bg, dtype="float32")

    # foreground

    # score1 = np.pad(score1, 1, 'constant', constant_values=m1)
    score1 = np.array(score1 * 255, dtype="uint8") # convert to uint8
    #score1 = cv2.bilateralFilter(score1,9,75,75)
    app1 = dahu.DahuApplication(score1)
    app1.setMarkers(fg_markers, bg_markers)
    fg1 = np.array(app1.fg, dtype="float32")
    bg1 = np.array(app1.bg, dtype="float32")


    fg = fg/255
    bg1 = bg1/255



    #### compute sigma

    neighbor = [[0,1],[0,-1],[1,0],[-1,0]]

    sigma1 = 0

    for x in range (h):
        for y in range (w):

            u = x*w + y
            for i in range (4):
                a = x + neighbor[i][0]
                b = y + neighbor[i][1]

                if (a >= 0 and a <h and b >= 0 and b < w):
                    v = a*w + b

                    if (v > u):
                        if sigma1 < eu_dis(LAB_map[x][y], LAB_map[a][b]):
                            sigma1 = eu_dis(LAB_map[x][y], LAB_map[a][b])





    sigma1 = sigma1 ** 2 * 1

    print("sigma1 = " + str(sigma1))


    ############ Graphcut

    lamda  = 0.2


    label = np.zeros((h,w))


    label[fg<=bg1] = 1

    oldEnergy = computeEnerge(label, fg, bg1,  neighbor, lamda,  LAB_map, sigma1)
    print(oldEnergy)

    # f = plt.figure(1)
    # plt.imshow(label)



    nodes = []
    edges = []

    cap_source = fg
    cap_sink = bg1




    for x in range (h):
        for y in range (w):
            u = x*w + y
            nodes.append((u, cap_source[x][y] , cap_sink[x][y]))

    # print(u, reflect.index(u))




    for x in range (h):
        for y in range (w):
            u = x*w + y        
            for i in range (4):
                a = x + neighbor[i][0]
                b = y + neighbor[i][1]
                if (a >= 0 and a <h and b >= 0 and b < w):
                    v = a*w + b
                    if (v > u):
                        weight = lamda * np.e ** (-(eu_dis(LAB_map[x][y], LAB_map[a][b]) ** 2) / sigma1)
                        edges.append((u, v, weight))    


    ####GraphCuts####
    g = maxflow.Graph[float](len(nodes), len(edges))

    nodelist = g.add_nodes(len(nodes))
    for node in nodes:
        g.add_tedge(node[0], node[1], node[2])

    for edge in edges:
        g.add_edge(edge[0], edge[1], edge[2], edge[2])

    flow = g.maxflow()  

    for vect in nodes:
        v = vect[0]
        if g.get_segment(v) == 0:  # beta
            x = int(np.floor(v/w))
            y = v%w
            label[x][y] = 0
        else:  # alpha
            label[x][y] = 1


    newEnergy = computeEnerge(label, fg, bg1,  neighbor, lamda,  LAB_map, sigma1)
    print(newEnergy)






    #  Post processing

    label_gray[label_gray>100] = 255

    label = np.array(label, dtype="uint8") # convert to uint8
    # Find largest contour in intermediate image so that it contains the markers 
    cnts, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    con = []
    for i in range(len(cnts)):
        biggest = np.zeros(label.shape, np.uint8)
        cv2.drawContours(biggest, [cnts[i]], -1, 255, cv2.FILLED)
        biggest = cv2.bitwise_and(label_gray, biggest)
        if (np.sum(biggest) > 0):
            con.append(cnts[i])

    print(len(con))

    biggest = np.zeros(label.shape, np.uint8)

    if (len(con) != 0):
        cnt = max(con, key=cv2.contourArea)

        # Output            
        cv2.drawContours(biggest, [cnt], -1, 255, cv2.FILLED)

    print(np.max(biggest))
    biggest = np.array(biggest, dtype="uint8") # convert to uint8
    output_name = output_post_folder + entry
    cv2.imwrite(output_name, biggest)

    label = np.array(label*255, dtype="uint8") # convert to uint8
    output_name = output_folder + entry
    cv2.imwrite(output_name, label)