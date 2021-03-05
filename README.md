# Covid-Detection-From-Xray-Images
Hello, 
This is a repository for detecting Covid-19 from X-ray images. We need three types of X-ray images. This detection is based on anteroposterior x-ray images. 

1. x ray images with COVID (anteroposterior pics)
2. x ray images without COVID (anteroposterior pics)
3. non x-ray images or non anteroposterior pics. 
 


step 1. To run my code, please, change the path in the CORONAXRAY.m file 
to the directory containing all the sample images. 

step 2. Please, change the ind_for_test and image_index in the MATLAB file 
to 30 from 270. This will cover all the sample X-ray images. I left some 
comments in the code.

step 3. It may take a while to run the code with all different k values.
Please, set k as 5 (k=5) for k-fold and comment out the for loop with the end. 
I left a tiny comment on this. 

If anything does not work, please, email me at johnkm37@gmail.com

To obtain sample X-ray images with COVID-19, I suggest you visit Dr. Joseph Cohen's Github website: 
https://github.com/ieee8023/covid-chestxray-dataset 

I appreciate you reading this. 
