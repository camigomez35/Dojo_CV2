import cv2
import numpy as np

im1 = cv2.imread('blob.jpg', cv2.IMREAD_GRAYSCALE)

im2 = cv2.imread('mugres.jpg', cv2.IMREAD_GRAYSCALE)

parametros = cv2.SimpleBlobDetector_Params()

## Cambiar umbrales
parametros.minThreshold = 10
parametros.maxThreshold = 500

## Filtrar por area
parametros.filterByArea = True
parametros.minArea = 100

## Filtrar por circularidad
parametros.filterByCircularity = True
parametros.minCircularity = 0.1

## Filtrar por canvexitividad
parametros.filterByConvexity = True
parametros.minConvexity = 0.9

## Filtrar por inercia
parametros.filterByInertia = True
parametros.minInertiaRatio = 0.01

## Crear detector con los parametros 
detector = cv2.SimpleBlobDetector_create(parametros)

## DEtector de manchas
puntosClave = detector.detect(im1)
puntosClave2 = detector.detect(im2)

## Dibujar circulos encima de las manchas
im_con_manchas = cv2.drawKeypoints(im1, puntosClave, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Blob', im_con_manchas)

cv2.waitKey(0)

im_con_manchas2 = cv2.drawKeypoints(im2, puntosClave2, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Manchas', im_con_manchas2)

cv2.waitKey(0)