# Self-Supervised Learning with Application for Infant Cerebellum Segmentation and Analysis
### Little is known about the early cerebellar development during the first two postnatal years, due to the challenging tissue segmentation caused by tightly-folded cortex, extremely low and dynamic tissue contrast, and large inter-site data heterogeneity. In this study, we propose a self-supervised learning (SSL) framework for infant cerebellum segmentation with multi-domain MR images.

<img src="https://github.com/YueSun814/Img-folder/blob/main/Github_framework.jpg" width="100%">

<font size=10> Figure 1. Source domain is 24-month-old subjects with manual labels and the target domain is unlabeled 18-month-old subjects. Our SSL framework consists of two steps. In Step 1, in the source domain, a segmentation model (i.e., ADU-Net) is trained based on a number of training subjects with manual labels, and then applied to testing subjects to automatically generate segmentations. Then, a confidence network is trained to evaluate the reliability of those automated segmentations at the voxel level. In Step 2, a set of reliable training samples from the testing subjects is automatically generated to train a segmentation model for the target domain accordingly, guided by a proposed spatially-weighted cross-entropy loss. </font>

## Method
We take advantage of accurate manual labels from 24-month-old cerebellums and transfer them to other time-points. To alleviate the domain shift issue, we propose to automatically generate a set of reliable training samples for target domains. To be specific, we first borrow the manual labels from 24-month-old cerebellums with high contrast (i.e., source domain) to train a segmentation model. Then the segmentation model is directly applied to testing imaging data from the target domain. We further utilize a confidence map to identify the reliability of automated segmentations and generate a set of reliable training samples for the target domain. Lastly, we train a target-domain-specific segmentation model with a spatially-weighted cross-entropy loss function, based on the generated reliable training samples. Experiments on three cohorts and one challenge demonstrate superior performance of our proposed framework.

## Data and MRI preprocessing
### Data

1. Public cerebellum data were from three cohorts, i.e., UNC/UMN Baby Connectome Project (BCP), Philips scans collected by Vanderbilt University, and National Database for Autism Research (NDAR). The data of BCP, Philips scans (Vanderbilt U) and NDAR were collected by 3T Siemens scanner, 3T Philips scanner and 3T Siemens scanner, respectively. 

    BCP: <https://nda.nih.gov/edit_collection.html?id=2848/>

    Philips scans (Vanderbilt U): <https://github.com/YueSun814/Philips_data.git/>

    NDAR (autism): <https://nda.nih.gov/edit_collection.html?id=19/>  


3. Public cerebrum data were provided by the iSeg-2019 challenge, in which multi-site data were collected by 3T Siemens and 3T GE scanners using different imaging parameters. 

    iSeg-2019: <https://iseg2019.web.unc.edu/>
    
### Data analysis

The image preprocessing steps, including skull stripping and extraction of the cerebellum, was performed by a public infant cerebrum-dedicated pipeline (iBEAT V2.0 Cloud, <http://www.ibeat.cloud>). The network was implemented using the Caffe deep learning framework (Caffe 1.0.0-rc3). The data testing used the custom Python code (Python 2.7.17). 

## Training and Testing
### File descriptions
> Training_subjects
>> 18 T1w MRIs with corresponding manual labels. 

>> ***subject-x-T1w.nii.gz***: the T1w MRI. 

>> ***subejct-x-label.nii.gz***: the manual label. 

> Testing_subjects

>> The subjects in the folder ***Testing_subjects*** are only randomly selected examples for the model testing.

>> ***subject-x-T1.hdr***: the T1w MRI. 

>> ***subject-x-T2.hdr***: the T2w MRI.  

> Template

>> ***Template_T1.hdr***: a template for histogram matching of T1w images.

>> ***Template_T2.hdr***: a template for histogram matching of T2w images.

> Dataset

>> The datasets in the folder ***Dataset*** are only randomly selected examples for the model training. Please download them from <https://github.com/YueSun814/SSL-dataset> firstly. 

>> ***subject-x-dataset-24m.hdf5***: a dataset for the 24-month-old segmentation model. 

>> ***subject-x-dataset-18m.hdf5***: a dataset for the 18-month-old segmentation model. 

>> ***subject-x-dataset-cp.hdf5***: a dataset for the confidence model. 

> Segmentation_model-24_month 
>> ***infant_train.prototxt***: the ADU-Net structure with a cross-entropy loss.

>> ***deploy.prototxt***: a duplicate of the train prototxt.

>> ***train_dataset.txt***: paths of training dataset.

>> ***test_dataset.txt***: paths of testing dataset.

>> ***segtissue.py***: the testing file.

> Segmentation_model-18_month 
>> ***infant_train.prototxt***: the ADU-Net structure with the proposed spatially-weighted cross-entropy loss.

>> ***deploy.prototxt***: a duplicate of the train prototxt.

>> ***train_dataset.txt***: paths of training dataset.

>> ***test_dataset.txt***: paths of testing dataset.

>> ***segtissue.py***: the testing file.

> Confidence_model
>> ***infant_train.prototxt***: the network structure with a binary cross-entropy loss.

>> ***deploy.prototxt***: a duplicate of the train prototxt.

>> ***train_dataset.txt***: paths of training dataset.

>> ***test_dataset.txt***: paths of testing dataset.

>> ***segtissue.py***: the testing file.

### Steps:
1. System requirements:

    Ubuntu 18.04.5
    
    Caffe version 1.0.0-rc3
    
    To make sure of consistency with our used version (e.g., including 3d convolution, WeightedSoftmaxWithLoss et al.), we strongly recommend installing _Caffe_ using our released ***caffe_rc3***. The installation steps are easy to perform without compilation procedure: 
    
    a. Download ***caffe_rc3*** and ***caffe_lib***.
    
    caffe_rc3: <https://github.com/YueSun814/caffe_rc3>
    
    caffe_lib: <https://github.com/YueSun814/caffe_lib>
    
    b. Add paths of _caffe_lib_, and _caffe_rc3/python_ to your _~/.bashrc_ file. For example, if the folders are saved in the home path, then add the following commands to the _~/.bashrc_ 
   
   `export LD_LIBRARY_PATH=~/Lib64:$LD_LIBRARY_PATH`
   
   `export PYTHONPATH=~/Caffe_rc3/python:$PATH`
    
    c. Test Caffe 
    
    `cd caffe_rc3/build/tools`
    
    `./caffe`
    
    Then, the screen will show:  
    
    <img src="https://github.com/YueSun814/Img-folder/blob/main/caffe_display.jpg" width="50%">
    
    Typical install time: few minutes. 

In folder: ***Segmentation_model-24_month***

2. Generate a training dataset (hdf5) for training a 24-month-old segmentation model: the intensity images and corresponding manual labels can be found in folder ***Training_subejcts***. 

    > Some training samples are prepared in the folder ***Datasets***.  

3. Training a segmentation model of 24 months old (_SegM-24_).

    `nohup ~/Caffe_rc3/build/tools/caffe train -solver solver.prototxt -gpu x >train.txt 2>&1 &     #submit a model training job using gpu x`
    
    Expected output: _iter_xxx.caffemodel/_iter_xxx.solverstate
    
    Trained model: SegM_24.caffemodel/SegM_24.solverstate

4. Using _SegM-24_ to derive automated segmentations and probability tissue maps for 24-month-old subjects by performing a 2-fold cross-validation. 

   > Performing histogram matching for testing subjects first with templates (in ***Template***).
   > The testing subjects in ***Testing_subjects*** have performed histogram matching. 

    `python segtissue.py`  
    
    Output folders: subject-1/subject-2
    
    Expected run time: about 1~2 minutes per subject.

In folder: ***Confidence_model***

5. Generate a training dataset (hdf5) for confidence model: computing confidence maps, defined as the differences between the manual labels (in folder ***Training_subejcts***) and the automated segmentations (_step 4_), is regarded as ground truth to train a confidence network. The automated segmentations and corresponding tissue probability maps (_step 4_) are used as inputs. 

    > Some training samples are prepared in the folder _Datasets_. 

6. Training a confidence model (_ConM_).

    `nohup ~/Caffe_rc3/build/tools/caffe train -solver solver.prototxt -gpu x >train.txt 2>&1 &     #submit a model training job using gpu x`
    
    Expected output: _iter_xxx.caffemodel/_iter_xxx.solverstate 
    
    Trained model: ConM.caffemodel/ConM.solverstate
  
In folder: ***Segmentation_model-24_month***  

7. Using _SegM-24_ to derive automated segmentations and probability tissue maps for **18**-month-old subjects.

In folder: ***Confidence_model***  

8. Using _ConM_ to evaluate the reliability of the automated segmentation (_step 7_) at each voxel.

    `python segtissue.py`  
    
    Output folders: subject-1/subject-2
    
    Expected run time: about 1~2 minutes per subject.

In folder: ***Segmentation_model-18_month*** 

9. Generate a training dataset (hdf5) for training a 18-month-old segmentation model (Inputs: intensity images and the confidence maps in _step 8_; target: automated segmentations in _step 7_).

    > Some training samples are prepared in the folder _Datasets_. 

10. Training a segmentation model of 18 months old (_SegM-18_).

    `nohup ~/Caffe_rc3/build/tools/caffe train -solver solver.prototxt -gpu x >train.txt 2>&1 &     #submit a model training job using gpu x`
    
    Expected output: _iter_xxx.caffemodel/_iter_xxx.solverstate
    
    Trained model: SegM_18.caffemodel/SegM_18.solverstate
    
11. Testing subjects.
    
    `python segtissue.py`  
    
    Output folders: subject-1/subject-2
    
    Expected run time: about 1~2 minutes per subject.


## License: *LICENSE.txt*

