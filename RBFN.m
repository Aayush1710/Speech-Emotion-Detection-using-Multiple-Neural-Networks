%%training
data=zeros(501,5*10);%matrix of 501*50 training
for i=1:10 %converting 5 voice samples in array in 1 iteration(happy,crying,excited,angry,neutral)
    no=num2str(i);
    emo1=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Train/','happy',no,'.wav');
    emo2=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Train/','crying',no,'.wav');
    emo3=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Train/','excited',no,'.wav');
    emo4=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Train/','angry',no,'.wav');
    emo5=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Train/','neutral',no,'.wav');
%% Feature Extraction using PCA
[b,~]=audioread(emo1);
d=fdesign.lowpass('Fp,Fst,Ap,Ast',0.15,0.25,1,60);
Hd=design(d,'equiripple');
b=filter(Hd,b);
[coeff,score]=pca(b);
data(2:end,5*i-4)=score(1:500);
data(1,5*i-4)=1;

[b,~]=audioread(emo2);
d=fdesign.lowpass('Fp,Fst,Ap,Ast',0.15,0.25,1,60);
Hd=design(d,'equiripple');
b=filter(Hd,b);
[coeff,score]=pca(b);
data(2:end,5*i-3)=score(1:500);
data(1,5*i-3)=1;

[b,~]=audioread(emo3);
d=fdesign.lowpass('Fp,Fst,Ap,Ast',0.15,0.25,1,60);
Hd=design(d,'equiripple');
b=filter(Hd,b);
[coeff,score]=pca(b);
data(2:end,5*i-2)=score(1:500);
data(1,5*i-2)=1;

[b,~]=audioread(emo4);
d=fdesign.lowpass('Fp,Fst,Ap,Ast',0.15,0.25,1,60);
Hd=design(d,'equiripple');
b=filter(Hd,b);
[coeff,score]=pca(b);
data(2:end,5*i-1)=score(1:500);
data(1,5*i-1)=1;

[b,~]=audioread(emo5);
d=fdesign.lowpass('Fp,Fst,Ap,Ast',0.15,0.25,1,60);
Hd=design(d,'equiripple');
b=filter(Hd,b);
[coeff,score]=pca(b);
data(2:end,5*i)=score(1:500);
data(1,5*i)=1;

end
% Target Matrix 

Target=repmat(eye(5),10,1)';
Input=data(2:end,:);

Input_train=Input(:,1:50);
Target_train=Target(:,1:50);

%%Training Network (error,spread factor,max neurons,step size)
RBF = newrb(Input_train,Target_train,0.0,0.5,500,9); % vary spread 



%%Testing
data=zeros(501,5*10);%data matrix of 501*50 for training
for i=1:10 %converting 5 voice samples in array in 1 iteration(happy,crying,excited,angry,neutral)
    no=num2str(i); 
    emo1=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Testing/','happy',no,'.wav');
    emo2=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Testing/','crying',no,'.wav');
    emo3=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Testing/','excited',no,'.wav');
    emo4=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Testing/','angry',no,'.wav');
    emo5=strcat('C:/Users/Sanjeev Patial/Desktop/AI/Testing/','neutral',no,'.wav');

%% Feature Extraction using PCA
[b,~]=audioread(emo1);
d=fdesign.lowpass('Fp,Fst,Ap,Ast',0.15,0.25,1,60);
Hd=design(d,'equiripple');
b=filter(Hd,b);
[coeff,score]=pca(b);
data(2:end,5*i-4)=score(1:500);
data(1,5*i-4)=1;

[b,~]=audioread(emo2);
d=fdesign.lowpass('Fp,Fst,Ap,Ast',0.15,0.25,1,60);
Hd=design(d,'equiripple');
b=filter(Hd,b);
[coeff,score]=pca(b);
data(2:end,5*i-3)=score(1:500);
data(1,5*i-3)=1;

[b,~]=audioread(emo3);
d=fdesign.lowpass('Fp,Fst,Ap,Ast',0.15,0.25,1,60);
Hd=design(d,'equiripple');
b=filter(Hd,b);
[coeff,score]=pca(b);
data(2:end,5*i-2)=score(1:500);
data(1,5*i-2)=1;

[b,~]=audioread(emo4);
d=fdesign.lowpass('Fp,Fst,Ap,Ast',0.15,0.25,1,60);
Hd=design(d,'equiripple');
b=filter(Hd,b);
[coeff,score]=pca(b);
data(2:end,5*i-1)=score(1:500);
data(1,5*i-1)=1;

[b,~]=audioread(emo5);
d=fdesign.lowpass('Fp,Fst,Ap,Ast',0.15,0.25,1,60);
Hd=design(d,'equiripple');
b=filter(Hd,b);
[coeff,score]=pca(b);
data(2:end,5*i)=score(1:500);
data(1,5*i)=1;

end
Target=repmat(eye(5),10,1)';
Input=data(2:end,:);

%%Test
%[Input_test,Target_test]=Collect_dat('Test');
Input_test=Input(:,1:50);
Target_test=Target(:,1:50);

%% 

y3=RBF(Input_test);

%Evaluation :  Confusion Matrix : Efficiency

Y = zeros(5, size(y3, 2));
[temp , ind] = max(y3, [],1);
for i = 1:length(Y)
Y(ind(i), i) = 1;
end
[c, cm, C] = confusion(Y,Target_test);
plotconfusion(Y,Target_test);
%% Display
fprintf('Accuracy = %2.2f %% \n', (1-c)*100);
fprintf('Confusion matrix')
display(cm')
