import SimpleITK as sitk
from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio
from scipy import ndimage as nd

# Make sure that caffe is on the python path:
# this is the path in GPU server ??? how to revise it ???

caffe_root = '~/caffe_rc3/'
import sys
sys.path.insert(0, caffe_root + 'python')
print (caffe_root + 'python')
import caffe
# very important, select GPU device
caffe.set_device(0)
# set gpu mode
caffe.set_mode_gpu()
# load the solver and create train and test nets
# ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
solver = None  
protopath=''
modelpath=''
mynet = caffe.Net(protopath+'deploy.prototxt',modelpath+'SegM_24.caffemodel',caffe.TEST)
print("blobs {}\nparams {}".format(mynet.blobs.keys(), mynet.params.keys()))

patch_r=32
d1=32
d2=32
d3=32
dFA=[d1,d2,d3] # size of patches of input data

dSeg=[d1,d2,d3] # size of pathes of label data
step1=8
step2=8
step3=8
step=[step1,step2,step3]

#the number of classes in this segmentation project
NumOfClass=4 #bg, wm, gm, csf

def find_boundary_img(Img, margin1, margin2, margin3):

    [Height, Wide, Z] = Img.shape
    for i in range (0, Height-1, 1):
        temp = Img[i,:,:]
        if sum(sum(temp[:]))>0:
            a = i
            break
        
    for i in range(Height-1, 0, -1):
        temp = Img[i,:,:]
        if sum(sum(Img[i,:,:]))>0:
            b = i
            break
    for i in range(0, Wide-1, 1):
        temp = Img[:,i,:]
        if (sum(sum(temp[:]))>0):
            c = i
            break
    
    for i in range(Wide-1, 0, -1):
        temp = Img[:,i,:]
        if (sum(sum(temp[:]))>0):
            dd = i
            break
        
    for i in range(0, Z-1, 1):
        temp = Img[:,:,i]
        if (sum(sum(temp[:]))>0):
            e = i
            break
    
    for i in range(Z-1, 0, -1):
        temp = Img[:,:,i]
        if (sum(sum(temp[:]))>0):
            f = i
            break
    
    if (a-margin1/2<=0):
        a = margin1+1

    if (c-margin2/2<=0):
        c = margin2+1

    if (e-margin3/2<=0):
        e = margin3+1

    if(b+margin1>=Height):
        b = Height-margin1

    if(dd+margin2>=Wide):
        dd = Wide-margin2

    if(f+margin3>=Z):
        f = Z-margin3
        
    return a,b,c,dd,e,f
    

def n4BiasCorr(img):
    #return img
    ori_type = img.GetPixelIDValue()
    corrector = sitk.N4BiasFieldCorrectionImageFilter();
    maskImage = sitk.OtsuThreshold(img,0,1,200)
    tmp = sitk.Cast(img,sitk.sitkFloat32)
    tmp = corrector.Execute(tmp, maskImage)
    tmp = sitk.Cast(tmp, ori_type)
    return tmp

def hist_match(img,temp):
    ''' histogram matching from img to temp '''
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    res = matcher.Execute(img,temp)
    return res

def convert_label(label_img):
    label_processed = np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice = label_img[:,:,i]
        label_slice[label_slice == 10] = 1
        label_slice[label_slice == 150] = 2
        label_slice[label_slice == 250] = 3
        label_processed[:,:,i] = label_slice
    return label_processed

def save_itk(image, subjectName):
#    itkimage = sitk.GetImageFromArray(image, isVector=False)
#    itkimage.SetSpacing(spacing)
#    itkimage.SetOrigin(origin)/shenlab/lab_stor6/suny/Experiments_atlas/testing_subjects/test
#    sitk.WriteImage(itkimage, filename, True) 
    
    data = image
    data = np.array(data)
    volout = sitk.GetImageFromArray(data)
    sitk.WriteImage(volout, '%s.img.gz' % subjectName)

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                  
		os.makedirs(path)            
		print("---  new folder...  ---")
		print("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")
   

def cropCubic(matT1,fileID,d,step,rate):
    eps=1e-5
    [row,col,leng]=matT1.shape
#    cubicCnt=0
        
    #print ('matT1 shape is ',matT1.shape)
    matT1Out = matT1
    matT1OutScale = nd.interpolation.zoom(matT1Out, zoom=rate)
    matSegScale=nd.interpolation.zoom(matT1Out, zoom=rate) #zoom in or zoom out

    matOut=np.zeros((matSegScale.shape[0],matSegScale.shape[1],matSegScale.shape[2],NumOfClass))
    PrOut=np.zeros((matSegScale.shape[0],matSegScale.shape[1],matSegScale.shape[2],NumOfClass))
    #print('Before loop of net')
    
    [aa,bb,cc,dd,ee,ff] = find_boundary_img(matT1, dFA[0], dFA[1], dFA[2])
    #aa=1
    print(aa,bb,cc,dd,ee,ff)
    #volMR=np.zeros((2,32,32,32))
    
    for i in range(aa,bb,step[0]):
        #print('come in i %s' % i)
        for j in range(cc,dd,step[1]):
            for k in range(ee,ff,step[2]):
                volMR = matT1[i:i+d[0],j:j+d[1],k:k+d[2]]  #patch 
                #volMR[1,:,:,:] = matT2[i:i+d[0],j:j+d[1],k:k+d[2]]  #patch   
                mynet.blobs['dataT1'].data[0,0,:,:,:]=volMR #put the patch into net
                mynet.forward()
                temppremat = mynet.blobs['softmax'].data[0].argmax(axis=0) #Note you have add softmax layer in deploy prototxt
                tempprob = mynet.blobs['softmax'].data[0] # probobility.

                for labelInd in range(NumOfClass): 
                    currLabelMat = np.where(temppremat==labelInd, 1, 0) #if satisfy the condition (labelInd), then output 1; else 0
                    matOut[i:i+d[0],j:j+d[1],k:k+d[2],labelInd]=matOut[i:i+d[0],j:j+d[1],k:k+d[2],labelInd]+currLabelMat
                    PrOut[i:i+d[0],j:j+d[1],k:k+d[2],labelInd]=PrOut[i:i+d[0],j:j+d[1],k:k+d[2],labelInd]+tempprob[labelInd,:,:,:]
    #print('end loop of net')
    sumOut=PrOut.sum(axis=3)
    
    PrOut0=PrOut[:,:,:,0]/(sumOut+eps)
    PrOut1=PrOut[:,:,:,1]/(sumOut+eps)
    PrOut2=PrOut[:,:,:,2]/(sumOut+eps)
    PrOut3=PrOut[:,:,:,3]/(sumOut+eps)

    matOut=matOut.argmax(axis=3) #always 3
    matOut=np.rint(matOut) #this line is necessary, it is very important, because it will convert datatype to make the nii.gz correct, otherwise, will appear strage shape
    return matOut,PrOut0,PrOut1,PrOut2,PrOut3
#this function is used to compute the dice ratio
def dice(im1, im2,tid):
    im1=im1==tid #make it boolean
    im2=im2==tid #make it boolean
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc

def main():
    
    dataPath = '../Testing_subjects/'
    monthList=['./']
    
    for mli in range(len(monthList)):
        
        ml= monthList[mli]
        data_path=os.path.join(dataPath,monthList[mli])
        items = os.listdir(data_path)
        newlist = []
        ids = set()
        for names in items:
            if names.endswith("-T1.hdr"):
            
               newlist.append(names)

        for f in newlist:
            ids.add(f.split('-T1.hdr')[0])
        ids = list(ids)
        print (ids)
        #for idn in range(1):
        for idn in range(len(ids)):
            print(idn)
            subject_name = ids[idn]
            print(subject_name)
            mkdir(ml+subject_name)

            f_Img1 = os.path.join(data_path,'%s-T1.hdr'%subject_name);
            img1_Org = sitk.ReadImage(f_Img1)
            imgT1=sitk.GetArrayFromImage(img1_Org)


            fileID='%02d'%idn
            rate=1
            imgT1=np.transpose(imgT1,(0,2,1))
            matOut,PrOut0,PrOut1,PrOut2,PrOut3=cropCubic(imgT1,fileID,dFA,step,rate)
            matOut=np.transpose(matOut,(0,2,1))
            PrOut0=np.transpose(PrOut0,(0,2,1))
            PrOut1=np.transpose(PrOut1,(0,2,1))
            PrOut2=np.transpose(PrOut2,(0,2,1))
            PrOut3=np.transpose(PrOut3,(0,2,1))
        
            volOut=sitk.GetImageFromArray(matOut)
            volPr0=sitk.GetImageFromArray(PrOut0)
            volPr1=sitk.GetImageFromArray(PrOut1)
            volPr2=sitk.GetImageFromArray(PrOut2)
            volPr3=sitk.GetImageFromArray(PrOut3)
        
        
            sitk.WriteImage(volOut,'./{}{}/{}-label.hdr'.format(ml, subject_name, subject_name))
            sitk.WriteImage(volPr0,'./{}{}/{}-Pr-BG.hdr'.format(ml, subject_name, subject_name))
            sitk.WriteImage(volPr1,'./{}{}/{}-Pr-CSF.hdr'.format(ml, subject_name, subject_name))
            sitk.WriteImage(volPr2,'./{}{}/{}-Pr-GM.hdr'.format(ml, subject_name, subject_name))
            sitk.WriteImage(volPr3,'./{}{}/{}-Pr-WM.hdr'.format(ml, subject_name, subject_name))                    

if __name__ == '__main__':     
    main()
