clear all;
close all;
clc;

%First, Loading the dataset 

% We need three sub-directories: one for COVID positive case and one for
% healthy cases, and one for X-ray images of different types. 

path = 'C:\Users\johnk\Desktop\School_Stuff\ELEC 421\project\Project_Source_code\XRAYImages' %directory path for all the X ray images

imds1 = imageDatastore(path,"LabelSource","foldernames",'IncludeSubfolders',true); %Manager for the collection of images


%We have three types of images: 
%1. x ray images with COVID (anteroposterior pics)
%2. x ray images without COVID (anteroposterior pics)
%3. non x-ray images or non anteroposterior pics. 

tbl = countEachLabel(imds1); %We see how many images of each label are contained in our subfolders

nets = {resnet50(), resnet101(),vgg16(),vgg19(),inceptionv3(),inceptionresnetv2(),xception()}; %array of different pretrained-models


net = inceptionv3(); 

%Current pretrained-models. We can change it to resnet50(), resnet101(),
%vgg19(),inceptionv3(),inceptionresnetv2(),xception()


Netsize = net.Layers(1).InputSize; %input size required by resnet50 network




TotalPredictionLabels = categorical({}); 
%For k fold cross-validation, we need to combine all the confusion
%matrices of each fold. Therefore, I am summing all the prediction labels
%which are categorical arrays.

TotalTestSetLabels = categorical({});

%For k fold cross-validation, we need to combine all the confusion
%matrices of each fold. Therefore, I am summing all the testset labels.


n = 1; %nth index number for k-fold
update = 1;%index for updating our index for test below.

accuracies = zeros(1,9); %initializing an array for the accuracies of our dectection
specificities = zeros(1,9); %initializing an array for the specificies of our dectection

sensitivities = zeros(1,9); %initializing an array for the sensitivities of our dectection

%The following loop generates k different folds
for k=2:10 %k for k-fold cross-validation. You can comment this for and the corresponding end and set k=5.
    
    while n <= k 

        ind_for_test = update:k:270;  %270 Can be changed to 30
        %This is an index for test set for our current fold.     
        %270 represents the total number of images. 
    
        subset_test = subset(imds1,ind_for_test); 
    
    %This is our current fold test set
    %subset function can only take indices less than 290. So that's why I
    %picked 270 picutres. 90 images for each case. 
    
        image_index = 1:1:270; %index for each image in our data. (Can be changed to 30)
        ind_for_training = setdiff(image_index,ind_for_test); 
    %By excluding all the test indices from the entire set of images, we can
    %find the index for traning. Setdiff function excludes them.
    
        subset_training = subset(imds1, ind_for_training);%These are our current fold training sets
    
    
        augmentedTraining = augmentedImageDatastore(Netsize, subset_training, 'ColorPreprocessing', 'gray2rgb'); %resize and convert our training set to the size required by the network

        augmentedTest = augmentedImageDatastore(Netsize, subset_test, 'ColorPreprocessing', 'gray2rgb'); %resize and convert our testing set to the size required by the network

        featureLayer = 'predictions'; 
    
    %Layer right before the classification layer
    %fc1000 for resnet50
    %fc1000 for resnet101
    %fc8 for vgg16
    %fc8 for vgg19
    %predictions for inceptionv3,inceptionresnetv2,xception
    
    
        trainingFeatures = activations(net, augmentedTraining, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
    %Extract features from our feature layer using the activations method.
    %Minibatch is set to 32 to make sure that GPU memory is good enough for the CNN and image data 
    %Activation output is in the form of columns. 


        trainingSetLabels = subset_training.Labels; %obtain traning set labels. 
    
        TestSetLabels = subset_test.Labels; %Testset labels. 


        classifier = fitcecoc(trainingFeatures, trainingSetLabels, 'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

    % fitcecoc trains a multiclass SVM classifier through a fast linear solver. 
    % We feed 'ObservationsIn' with 'columns' to match the arrangement required for training
    % features.
    
        testFeatures = activations(net, augmentedTest, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns'); 
    %We obtain test features through the CNN. 


        predictionLabel = predict(classifier, testFeatures, 'ObservationsIn', 'columns'); 
    %We feed our classifier with our testfeatures (passing CNN images features
    %to classfier). We try to predict responses from our testfeatures by using
    %our classifier. 


        TotalTestSetLabels = [TotalTestSetLabels; TestSetLabels]; %concatenating our layer categorical array
        TotalPredictionLabels = [TotalPredictionLabels; predictionLabel]; %concatenating our layer categorical array
    

        n = n+1;
        update = update+1;

    
    
    end
    

    %figure(k);
    
    %plotconfusion(TotalTestSetLabels , TotalPredictionLabels); %Plotting
    %confusion matrix
    
    %title(['Confusion matrix with ', num2str(k), ' fold cross-validation']);
    
    C = confusionmat(TotalTestSetLabels, TotalPredictionLabels)
    %We create a confusion matrix based on our results. 

    %The following outcome information is for COVID X-RAY AP Images
    TP = C(1,1); %True positive
    TN = C(2,2) + C(2,3) + C(3,2) + C(3,3); %True negative
    FN = C(1,2) + C(1,3); %False negative
    FP = C(2,1) + C(3,1); %Fasle positive 

    Sensitivity = (TP)/(TP+FN); %Sensitivity for this particular k-fold 
    sensitivities(k) = Sensitivity; %updating sensitivities 

    Specificity = (TN)/(TN+FP); %Specificity for this particular k-fold 
    specificities(k) = Specificity; %updating specificities 

    
    Accuracy = (TN + TP) / (TN + TP + FN + FP); %Accuracy for this particular k-fold 
    accuracies(k) = Accuracy; %updating accuracies.

    TotalPredictionLabels = categorical({}); 
    %Restoring total prediction labels to empty array

    TotalTestSetLabels = categorical({});     
    %Restoring total prediction labels to empty array

    n=1; %Restoring n to 1 before changing to a different k-fold 
    update = 1; %Restoring update to 1 before changing to a different k-fold

end 

