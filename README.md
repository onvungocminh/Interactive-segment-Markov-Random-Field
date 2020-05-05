# Interactive-segment-Markov-Random-Field
Interactive segmentation using the Dahu distance and Markov Random Field

In this project, I want to present the Dahu distance and its application in interactive segmentation.
Specifically, the Dahu distance is used to compute the confidence map, in which, values of pixels
inside the object are higher than pixels outside the object.
Also we combine this confidence map with the smoothness term, which performs the similarity between 
neighbor pixels. These two terms are integrated in the Markov Random Field model to optimize the
energy function.

The GUI is writen in demo_Dahu_MRF.py.
