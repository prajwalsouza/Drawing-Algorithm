# All imports

import numpy as np
import cv2
import time

'''
Here are the important values to be set.
'''

#The Number of Sectors
xsectors = 2
ysectors = 2


#The Upper and lower thresholds
upperthreshold = 120
lowerthreshold = 0



#Resize image?
#If resizebyfactor : 3, autoresize : 2, regularresize : 1, No : 0
resizeimg = 0




#Applicable only if autoresize is allowed.

#---------------------------------------------

#Modes
portrait = 1  #If yes : 1, No : 0

#Maximum Width (In portrait)
mwidth = 210
#Maximum Height
mheight = 297

#Rotation
counterclockwise = False

#--------------------------------------------


Athreshold = 0




#Applicable only if regularesize is allowed.

#---------------------------------------------

#Resize to :
rwidth = 210
rheight = 297

#--------------------------------------------
	
#Applicable only if resizing by factor is allowed.

#---------------------------------------------

#Resize to :
xfactor = 2
yfactor = 2

#--------------------------------------------

  
prefer = 0

timer1 = time.time()

def getpoints(y2,x2,y1,x1,operation):
	global prefer

	#directionvector
	thepoint = np.array([[y2],[x2]])
	directionvector = np.array([[y2-y1],[x2-x1]])
	
	if(operation == 4):
		#Point in the directiion
		pa1 = directionvector + thepoint
		sendpoint = pa1


	if(operation == 3):
		#Points forward diagonal : pb1 and pb2
		a = np.array([[ 1, 1], [ -1, 1]])
		b = a.dot(directionvector)
		tpb1 = b
		k = np.linalg.norm(b)
	
		if(k >= 2):
			tpb1 = np.multiply(0.5,b)
		pb1 = tpb1 + thepoint
		sendpoint = pb1
		

	if(operation == 5):
		a = np.array([[ 1, -1], [ 1, 1]])
		b = a.dot(directionvector)
		tpb2 = b
		k = np.linalg.norm(b)

		if(k >= 2):
			tpb2 = np.multiply(0.5,b)
		pb2 = tpb2 + thepoint
		sendpoint = pb2



	if(operation == 2):
		#Points to the right and left
		a = np.array([[ 0, 1], [ -1, 0]])
		b = a.dot(directionvector)
		tpc1 = b
		k = np.linalg.norm(b)

		if(k >= 2):
			tpc1 = np.multiply(0.5,b)
		pc1 = tpc1 + thepoint
		sendpoint = pc1

	if(operation == 6):
		a = np.array([[ 0, -1], [ 1, 0]])
		b = a.dot(directionvector)
		tpc2 = b
		k = np.linalg.norm(b)
	
		if(k >= 2):
			tpc2 = np.multiply(0.5,b)
		pc2 = tpc2 + thepoint
		sendpoint = pc2



	if(operation == 1):
		#Points backward diagonal
		angle135 = 3*(np.pi)/4
		a = np.array([[ -1, 1], [ (-1), -1]])
		b = a.dot(directionvector)
		tpd1 = b
		k = np.linalg.norm(b)
	
		if(k >= 2):
			tpd1 = np.multiply(0.5,b)
		pd1 = tpd1 + thepoint
		sendpoint = pd1

	if(operation == 7):
		a = np.array([[ -1, -1], [ 1, -1]])
		b = a.dot(directionvector)
		tpd2 = b
		k = np.linalg.norm(b)
	
		if(k >= 2):
			tpd2 = np.multiply(0.5,b)
		pd2 = tpd2 + thepoint
		sendpoint = pd2


	return sendpoint


preferencetray = [[4,3,5,2,6,1,7],[1,2,3,4,5,6,7],[7,6,5,4,3,2,1]]

def shaderpreference(counter):
	global prefer, preferencetray
	counter = counter - 1
	prefercount = preferencetray[prefer][counter]

	return prefercount




#finding the blackpixel
def findnextblack(y2,x2,y1,x1):
	global strokenum, stroke, startingy, startingx, secheight, secwidth, bpoints
	global prefer
	flag = 1
	opcounts = 1
	while(flag == 1 and opcounts <= 7):
		topcounts = shaderpreference(opcounts)
		nearbypoint = getpoints(y2,x2,y1,x1,topcounts)
		ny = int(nearbypoint[0])
		nx = int(nearbypoint[1])

		#blacks = 1

		#if(ny < img.shape[0] and nx < img.shape[1]):
		#    blacks = img[ny,nx] #Not being racist here.



		#Checking if the point is in the sector

		if(ny < startingy  or ny > secheight or nx < startingx or nx > secwidth):
			k = 2

		#if(k == 0):
		#    storeinImemory(ny,nx)

		v = checkbpoint(ny,nx)
#        print(v)
#        print(nearbypoint)
#        print(bpoints)
#        time.sleep(1)
		
		
		if(v == 1):
			flag = 0
			blackpoint = [ny,nx]
			bpoints.remove(blackpoint)
			if(prefer == 0):
				if(topcounts <= 3):
					aprefer = 1
				if(topcounts > 4):
					aprefer = 2
				if(topcounts == 4):
					aprefer = 0
			if(prefer == 1):
				if(topcounts < 4):
					aprefer = 1
				if(topcounts == 4):
					aprefer = 0
				if(topcounts > 4):
					aprefer = 2
			if(prefer == 2):
				if(topcounts < 4):
					aprefer = 1
				if(topcounts == 4):
					aprefer = 0
				if(topcounts > 4):
					aprefer = 2
			prefer = aprefer
				
			
		
		opcounts = opcounts + 1

	if(flag == 1):
		blackpoint = [123456789123456789, 123456789123456789]

	return blackpoint




	









#Analyzing Sectors/Creating the iteration memory

imemory = []


#Stroke memory and stroke number
strokememory = []
strokenum = 0

stroke = []

#Stroke data of a stroke




 




#Store in imemory
def storeinImemory(spy,spx):
	imemory.append([spy,spx])


	



#Check if present in imemory
def checkbpoint(py,px):
	global bpoints
	iterator = 0
	inmemory = 0
	while(iterator < len(bpoints) and inmemory == 0):
		my, mx = bpoints[iterator]
		if(my == py and mx == px):
			inmemory = 1
		else:
			inmemory = 0
	
		iterator = iterator + 1
		
	return inmemory


#Find starting points
def startingpointofasector(s1):
	global xsectors, ysectors, sectors
	if((s1 % xsectors) == 0 and s1 != xsectors):
		r1 = s1 - 1
		sheight = sectors[r1][0] + 1
		swidth = 0
	if(s1 > xsectors and (s1 % xsectors) != 0):
		r1 = s1 - xsectors - 1
		sheight = sectors[r1][0] + 1
		swidth = sectors[r1][1] + 1
	if(s1 == xsectors):
		r1 = s1 - 1
		sheight = sectors[r1][0] + 1
		swidth = 0
	if(s1 < xsectors and s1 != 0):
		r1 = s1 - 1
		sheight = 0
		swidth = sectors[r1][1] + 1
	if(s1 == 0):
		sheight = 0
		swidth = 0
	sectordim = [sheight,swidth]
	return sectordim
		











#New approach

def analyzeSector(s):
	global strokenum, stroke, startingy, startingx, secheight, secwidth, sectorblackcollection, bpoints, strokeendpoint
	bpoints = sectorblackcollection[s]
	secheight = sectors[s][0]
	secwidth = sectors[s][1]
	startingpoints = startingpointofasector(s)
	startingy = startingpoints[0]
	startingx = startingpoints[1]
	if(len(bpoints) != 0):
		strokeendpoint = bpoints[0]
	while(len(bpoints) != 0):
		points = pointfinder(strokeendpoint)
		i = points[0]
		j = points[1]
		v = checkbpoint(i,j)
		if(v == 1):
			bpoints.remove([i,j])
		strokenum = strokenum + 1
		stroke.append([i,j])
		startp = [i,j]
		nextpoint = [1,1]
		mainpointy = i
		mainpointx = j
		prevxpix = j - 1
		prevypix = i
		while(nextpoint[0] != 123456789123456789):
			nextpoint = findnextblack(mainpointy,mainpointx,prevypix,prevxpix)
			if(nextpoint[0] != 123456789123456789):
				nextpointy = nextpoint[0]
				nextpointx = nextpoint[1]
				prevypix = mainpointy
				prevxpix = mainpointx
				mainpointy = nextpointy
				mainpointx = nextpointx
				stroke.append([nextpointy,nextpointx])
				strokeendpoint = [mainpointy,mainpointx]
		strokelength = len(stroke)
		strokeendpoint = [mainpointy,mainpointx]
		if(strokelength == 1):
			strokedata = stroke
			strokememory.append([strokenum,s,startp,startp,strokelength,strokedata])
			stroke = []
		if(strokelength > 1):
			endp = [mainpointy,mainpointx]
			strokedata = stroke
			strokememory.append([strokenum,s,startp,endp,strokelength,strokedata])
			stroke = []
	
	
def pointfinder(epoint):
	global bpoints
	ey = epoint[0]
	ex = epoint[1]
	if(epoint == bpoints[0]):
		pointfound = epoint
	if(epoint != bpoints[0]):
		pointfound = surroundingpointfinder(epoint)

	return pointfound
			
			

movements = [[0,0], [0,1], [-1,0], [0,-1], [1,0]]




def surroundingpointfinder(points1):
	points1 = np.array(points1)
	flag7 = 1
	iter1 = 1
	while(flag7 == 1):
		vals = checkforN(iter1)
		thedirect = movements[vals]
		thedirect = np.array(thedirect)
		if(iter1 == 1):
			theneighbour = thedirect + points1
		if(iter1 != 1):
			theneighbour = thedirect + theneighbour
		dy = theneighbour[0]
		dx = theneighbour[1]
		v = checkbpoint(dy,dx)
		if(v == 1):
			flag7 = 0
		iter1 = iter1 + 1

	return theneighbour
		




def checkforN(n1):
	flag9 = 1
	k2 = 1
	while(flag9 == 1):
		m1 = k2*(k2+1)
		k2 = k2 + 1
		if(m1 >= n1):
			flag9 = 0
			k2 = k2 - 1
	c1 = m1 - k2
	if(n1 <= c1):
		thearvalue = 3
		if((k2 % 2) == 1):
			thearvalue = 1
	if(n1 > c1):
		thearvalue = 4
		if((k2 % 2) == 1):
			thearvalue = 2

	return thearvalue     
	
					
#The full image analysis
def analyzeimage():
	global strokenum
	for i in range(0,len(sectors)):
		strokenum = 0
		print("Sector %d analysis in progress." % i)
		analyzeSector(i)
		print("%d strokes were found in this sector." % strokenum)
				




#Sector divisions
					
def sectordivision():
	global sectors,  xsectors, ysectors
	yseclen = img.shape[0] / (ysectors)
	xseclen = img.shape[1] / (xsectors)
	sectors = []
	for i in range(1,ysectors):
		for j in range(1,xsectors):
			sectors.append([i*yseclen,j*xseclen])
			if(j == (xsectors - 1)):
				sectors.append([i*yseclen,img.shape[1]])
		if(i == (ysectors - 1)):
			for j in range(1,xsectors):
				sectors.append([img.shape[0],j*xseclen])
				if(j == (xsectors - 1)):
					sectors.append([img.shape[0],img.shape[1]])



	
		






#Converting Graystyle to BW.

def imageToBWandSectorData():
	global strokenum, stroke, startingy, startingx, secheight, secwidth, sectorblackcollection, upperthreshold, lowerthreshold
	sectorblackcollection = []
	for s in range(0,len(sectors)):
		secheight = sectors[s][0]
		secwidth = sectors[s][1]
		startingpoints = startingpointofasector(s)
		startingy = startingpoints[0]
		startingx = startingpoints[1]
		sectorblack = []
		for ypixel in range(startingy,secheight+1):
			for xpixel in range(startingx,secwidth+1):
				if(ypixel < img.shape[0] and xpixel < img.shape[1]):
					blackvalue = img[ypixel,xpixel]
					if(blackvalue < lowerthreshold):
						img[ypixel,xpixel] = 255
					if(blackvalue >= lowerthreshold and blackvalue <= upperthreshold):
						img[ypixel,xpixel] = 0
						sectorblack.append([ypixel,xpixel])
					if(blackvalue > upperthreshold):
						img[ypixel,xpixel] = 255
		sectorblackcollection.append(sectorblack)


print("This algorithm is a shader algorithm. It will try to draw the image moving in a zig-zag manner giving a shading effect.")
print("\nAnalysis Conditions for the Images : ")
print("The image will be divided into %d sectors horizontally." % xsectors)
print("The image will be divided into %d sectors vertically." % ysectors)

# print("")
print("If the pixel value in the image is in between %d and %d it will be considered as black area. Rest will be labelled white area." % (lowerthreshold,upperthreshold))

if(resizeimg == 0):
	print("The Image won't be resized.")

if(resizeimg == 1):
	print("The Image will be resized to width : %d and height : %d." % (rwidth,rheight))

if(portrait == 0):
	mwidth = 297
	mheight = 210

if(resizeimg == 2):
	if(portrait == 1):
		print("The Image will also be converted to a Portrait and resized to width : %d and height : %d." % (mwidth,mheight))
	if(portrait == 0):
		print("The Image will also be converted to a Landscape and resized to width : %d and height : %d." % (mwidth,mheight))



print("\nPlease select the Image file. \nIt's preferred to avoid high resolution images as the algorithm might take a lot of time. \nVector art type images are preferred.")

#opening the file
from Tkinter import Tk
from tkFileDialog import askopenfilename

Tk().withdraw()
filename = askopenfilename()

filenameparts = filename.split("/")
filenamelength = len(filenameparts)
filenameonly = filenameparts[filenamelength - 1]
filenamefull = filenameonly.split(".")
filenameonly = filenamefull[0]
filenameextension = filenamefull[1]

print("Selected File Path : ")
print(filename)



# Bring the image in graystyle
originalimg = cv2.imread(filename,0)



#Resizing the Image

angleturn = -90

if(counterclockwise == True):
	angleturn = 90

'''

code taken from
http://stackoverflow.com/questions/11764575/python-2-7-3-opencv-2-4-after-rotation-window-doesnt-fit-image

'''


def rotateAndScale(img, scaleFactor, degreesCCW):
	oldY = img.shape[0]
	oldX = img.shape[1]#note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
	M = cv2.getRotationMatrix2D(center=(oldX/2,oldY/2), angle=degreesCCW, scale=scaleFactor) #rotate about center of image.

	#choose a new image size.
	newX,newY = oldX*scaleFactor,oldY*scaleFactor
	#include this if you want to prevent corners being cut off
	r = np.deg2rad(degreesCCW)
	newX,newY = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))

	#the warpAffine function call, below, basically works like this:
	# 1. apply the M transformation on each pixel of the original image
	# 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

	#So I will find the translation that moves the result to the center of that region.
	(tx,ty) = ((newX-oldX)/2,(newY-oldY)/2)
	M[0,2] += tx #third column of matrix holds translation, which takes effect after rotation.
	M[1,2] += ty

	rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX),int(newY)))
	return rotatedImg


if(resizeimg == 3):
	aheight, awidth = originalimg.shape[:2]
	img = cv2.resize(originalimg,((int(float(awidth)/float(xfactor))), (int(float(aheight)/float(yfactor)))), interpolation = cv2.INTER_CUBIC)
	print(" \nImage Resized : 3 ")
	print(img.shape[:2])


if(resizeimg == 1):
	img = cv2.resize(originalimg,(rwidth, rheight), interpolation = cv2.INTER_CUBIC)
	print(" \nImage Resized : 1")


if(resizeimg == 2):
	aheight, awidth = originalimg.shape[:2]
	if(portrait == 1):
		if(awidth >= aheight):
			originalimg = rotateAndScale(originalimg,1, angleturn)
		aheight, awidth = originalimg.shape[:2]
		wtoh = float(float(awidth)/float(aheight))
		nwidth = int(mheight * wtoh)
		nheight = mheight
		img = cv2.resize(originalimg,(nwidth, nheight), interpolation = cv2.INTER_CUBIC)
	if(portrait == 0):
		if(awidth <= aheight):
			originalimg = rotateAndScale(originalimg,1, angleturn)
		aheight, awidth = originalimg.shape[:2]
		wtoh = float(float(awidth)/float(aheight))
		nheight = int(mwidth / wtoh)
		nwidth = mwidth
		img = cv2.resize(originalimg,(nwidth, nheight), interpolation = cv2.INTER_CUBIC)
	print("\nImage Resized : 2")
	 


#Showing image after resize
if(resizeimg != 2 and resizeimg  != 1 and resizeimg != 3):
	img = originalimg
	# print("\nDisplaying Resized Image.")

print("Displaying the Image Selected. The Image will be divided into sectors for analysis.")
cv2.imshow('image',img)
k = cv2.waitKey(0)


# print(img.shape[:2])


print("\nSector Division Begins. ")
#sectors division
sectordivision()
print("Sector Division Done. ")


#blur = cv2.GaussianBlur(img,(5,5),0)
#img = cv2.threshold(blur,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
#img = cv2.GaussianBlur(img,(1,1),0)
if(Athreshold == 1):
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	print("\nAdaptive Thresholding was applied.")
	print("Showing the image again after adaptive Thresholding.")
	cv2.imshow('image',img)
	k = cv2.waitKey(0)


print("\nConverting Image to Black and White regions.")
#Coverting image
imageToBWandSectorData()

print("Displaying Converted Image.")

# Displaying image for checks
cv2.imshow('image',img)
k = cv2.waitKey(0)



print("\nImage Analysis Begins. ")
print("There are %d sectors. Each of them will be analyzed." % len(sectors))

#Analysis of the full image
analyzeimage()

print("\nImage Analysis Done.")

#time calculations
timer2 = time.time()

totaltime = timer2 - timer1
totalhours = int(totaltime/3600)
totalmins = int(totaltime/60)

extramins = totalmins
extrasec = totaltime

if(totalhours != 0):
	extramins = totalmins % totalhours

if(totalmins != 0):
	extrasec = totaltime % totalmins
	
print("The Time taken by the program is %f seconds." % totaltime)
print("which is about %d hours, %d minutes and %d seconds." %(totalhours,extramins,extrasec))

print("\nThe Image and the associated data will be stored in a folder.")


import os
currentdir = str(os.getcwd())


directory = currentdir + "\\" + filenameonly

if not os.path.exists(directory):
	os.makedirs(directory)

import shutil

shutil.copy2(filename, directory)


convertedfile = directory + "\\" + 'OutputImage' + ' - ' + filenameonly + '.' + filenameextension
print("The expected output image is stored at the following path.")
print(convertedfile)
cv2.imwrite(convertedfile,img)

strokestart = ["S","S"]
strokeend = ["E","E"]

vdata = []

strokedata = strokememory
print("\nThe Stroke data will be converted to vector data.")

for i in range(0,len(strokedata)):
	k = strokedata[i]
	m = k[5]
	for j in range(0, len(m)):
		if(j == 0):
			n = m[j]
			vdata.append(n)
			vdata.append(strokestart)
		n = m[j]
		vdata.append(n)
	vdata.append(strokeend)


'''

Comparing vdata with blackpoints memory

print("Computing missing points.")
missing = []

for i in range(0,len(blackpixels)):
	c = blackpixels[i]
	flag1 = 0
	j = 0
	while(j < len(vdata) and flag1 == 0):
		if(c == vdata[j]):
			flag1 = 1
			missing.append(c)
		j = j + 1

missingnumber = len(blackpixels) - len(missing)

print("The number of missing points is %d." % missingnumber)
		
'''    

import pickle

print("Saving the vector data in image folder.")

dumphere = directory + "\\" + 'vectordata.p'
pickle.dump(vdata, open(dumphere, "wb" ))

import pickle
import numpy as np
#We move using the vector data

import time

#opening the file
from Tkinter import Tk
from tkFileDialog import askopenfilename

# Tk().withdraw()
# filename = askopenfilename()


vdata = pickle.load(open(dumphere, "rb" ))
svdata = pickle.load(open(dumphere, "rb" ))

startindicator = 0
startpoint = 0

print("\nThe vector data will be converted to a more compressed form and saved in the image folder.")


def recorddecider(ay,ax,by,bx,cy,cx):
	ya = cy - by
	yb = by - ay
	xa = cx - bx
	xb = bx - ax
	reply = 1
	if(yb == ya and xa == xb):
		reply = 0
	
	return reply


for i in range(0,len(vdata)):
	k = vdata[i]
	ky = k[0]
	kx = k[1]
	if(i > 2 and i < (len(vdata) - 1)):
		j = vdata[i-1]
		l = vdata[i+1]
		jy = j[0]
		jx = j[1]
		ly = l[0]
		lx = l[1]
		if(ky != 'E' and jy != 'E' and ly != 'E' and ky != 'S' and jy != 'S' and ly != 'S'):
			decision = recorddecider(jy,jx,ky,kx,ly,lx)
			if(decision == 0):
				svdata.remove([ky,kx])



print("The number of vectors were reduced from %d to %d." % (len(vdata), len(svdata)))

filenameparts = dumphere.split("/")
filenamelength = len(filenameparts)
filepath = ''
for s in range(0,filenamelength - 1):
	filepath = filepath + filenameparts[s] + "\\"

dumppath = directory + "\\" + 'compressed vectordata.p'
pickle.dump(svdata, open(dumppath, "wb" ))

#sending the data to arduino


'''

Pattern :

movecode-xdirection-xvalue-ydirection-yvalue

Movecode : 1 - pen down, 2 - pen up, 0 - move 
Direction : 1 - positive, 0 - negative

Eg: 0-1-0001-0-0002


'''


import pickle
#We move using the vector data
import pyautogui as pag
import time

#opening the file
from Tkinter import Tk
from tkFileDialog import askopenfilename

# Tk().withdraw()
# filename = askopenfilename()

pag.MINIMUM_DURATION = 0


print("\nThe Image can now be drawn on paint. Type 's' if you want to draw the image on Paint. Else type 'e' and press 'enter'.")
todraw = raw_input()
if todraw == 's':
	print("Open MS Paint. Select the appropriate pencil/brush tool. Then place the cursor in the Paint Window.")
	vdata = pickle.load(open(dumppath, "rb" ))

	resizex = 1 #Only positive integers
	resizey = 1


	print("You have 10 seconds from now to place the cursor. After that the mouse starts drawing the image.")
	time.sleep(10)

	print("The mouse starts to draw from now. Incase you find problems here, move the cursor to the left-top corner of the screen. The program will exit with an error.")
	timer1 = time.time()


	for i in range(0,len(vdata)):
		xdisplace = 100
		ydisplace = 200
		k = vdata[i]
		y = k[0]
		x = k[1]
		if(y == 'S' and i != 0):
			pag.mouseDown()
		if(y == 'E'):
			pag.mouseUp()
		if(y != 'S' and y != 'E'):
			x = (resizex)*x + xdisplace
			y = (resizey)*y + ydisplace
			pag.moveTo(x,y)
		
	print("Drawing finished.")
	timer2 = time.time()

	totaltime = timer2 - timer1

	totalhours = int(totaltime/3600)
	totalmins = int(totaltime/60)

	extramins = totalmins
	extrasec = totaltime

	if(totalhours != 0):
		extramins = totalmins % totalhours

	if(totalmins != 0):
		extrasec = totaltime % totalmins
		
	print("The Time taken by the program is %f." % totaltime)
	print("which is about %d hours, %d minutes and %d seconds." %(totalhours,extramins,extrasec))
	print("\nThe Image is drawn in MS Paint and saved in folder with the vector data.")
	time.sleep(2)
else:
	print("Thank you. The Image is saved in folder with the vector data.")
	time.sleep(2)





		



