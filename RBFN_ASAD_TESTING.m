%data collection
%%DATA COLLECTION TRAINING
data=zeros(501,5*10);%data matrix of 501*50 for training
for i=1:10 %converting 5 voice samples in array in 1 iteration(happy,crying,excited,angry,neutral)
    no=num2str(i); 
    name1=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Testing/','happy',no,'.wav');
    name2=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Testing/','crying',no,'.wav');
    name3=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Testing/','excited',no,'.wav');
    name4=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Testing/','angry',no,'.wav');
    name5=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Testing/','neutral',no,'.wav');

%% Feature Extraction using PCA
[b,~]=audioread(name1);
b=downsample(b,32);
[~,score]=pca(b);
data(2:end,5*i-4)=score(1:500);
data(1,5*i-4)=1;

[b,~]=audioread(name2);
b=downsample(b,32);
[coeff,score]=pca(b);
data(2:end,5*i-3)=score(1:500);
data(1,5*i-3)=1;

[b,~]=audioread(name3);
b=downsample(b,32);
[~,score]=pca(b);
data(2:end,5*i-2)=score(1:500);
data(1,5*i-2)=1;

[b,~]=audioread(name4);
b=downsample(b,32);
[~,score]=pca(b);
data(2:end,5*i-1)=score(1:500);
data(1,5*i-1)=1;

[b,~]=audioread(name5);
b=downsample(b,32);
[~,score]=pca(b);
data(2:end,5*i)=score(1:500);
data(1,5*i)=1;

end
Target=repmat(eye(5),10,1)';
Input=data(2:end,:);
%% TESTING

%[Input_test,Target_test]=Collect_dat('Test');
Input_test=Input(:,1:50);
Target_test=Target(:,1:50);

%% 
load RBF
y3=RBF(Input_test);

%Evaluation :  Confusion Matrix : Efficiency

Y = zeros(3, size(y3, 2));
[temp , ind] = max(y3, [],1);
for i = 1:length(Y)
Y(ind(i), i) = 1;
end
[c, cm, C] = confusion(Y,Target_test);

%% Display
fprintf('Accuracy = %2.2f %% \n', (1-c)*100);
fprintf('Confusion matrix')
display(cm')