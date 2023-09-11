# image Predictor
* A simple Image predictior which takes in a single image a produces the models best 3 predictions based on this image class ['Plane','Car', 'Bird','Cat','Deer','Dog','Frog','Forest','Ship','Truck'].
* it tends to run best on images with transparent backgrounds

## Description
Developed an Image Predictor Application using machine learning, trained on 50,000  images using a CiFAR 10 Data Set with a 65% prediction accuracy. 

## pre-requisite requirements:
Install the following modules: Pandas, tensorflow, tkinter, os ,numpy, keras, csv as modules using 'pip install [module name]'

## Images (Image used, prediction and heatmap)
![image](https://github.com/flashdash101/New-Image-Predict/assets/97402685/3de17ec7-c22e-48ac-af15-e2c627726147) ![image](https://github.com/flashdash101/New-Image-Predict/assets/97402685/c76b4786-48c0-4484-8c98-0f6d2d30620f) ![image](https://github.com/flashdash101/New-Image-Predict/assets/97402685/ea1bad14-dcf0-4360-9362-d323932e49ee)

![image](https://github.com/flashdash101/New-Image-Predict/assets/97402685/18737e24-c546-47e4-a8bb-f408b73cbc87) ![image](https://github.com/flashdash101/New-Image-Predict/assets/97402685/e8d1b2c7-2695-479a-9fb7-7294eaece3c8)![image](https://github.com/flashdash101/New-Image-Predict/assets/97402685/c46dd5f8-708c-400d-975e-5dac3b95a71b)

## How to run
* Assuming you have vscode just use the 'Run' Button.
* If not open terminal and use 'cd' to change the file directory to the directory of your download.
* Run 'Project3.py' 


## Issues
* Heavily affected by noise, if there is too much noise around the image you want to preict e.g. grass. It tends to focus on that area rather than the actual thing you want to predict. here is an example:
* ![image](https://github.com/flashdash101/New-Image-Predict/assets/97402685/2dcdcfdd-9cde-47ce-b6a4-82bca046bb54) ![image](https://github.com/flashdash101/New-Image-Predict/assets/97402685/a4619b2c-5fc8-4f37-997a-130bce321888) ![image](https://github.com/flashdash101/New-Image-Predict/assets/97402685/0003a91b-e6ca-429f-a7e3-2af84897d360)
* As you can see, the model thinks the image is a 'frog' given the green grass around the deer. 


## Future Ideas
* Utilize retraining, i am working on a feature which asks the user for input on whether the prediction was correct or not and this allows for future re-training (code is being worked on which is currently commented out in 'Project 3'.
* Utilizing data augmentation techniques when retraining to improve model accuracy.
* Utilizing Image Segmentation for better classification.
* Automatic Model retraining.
* Using a cloud base to store misclassied images.


