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

%%%
target_test = zeros(50,1);
Input_train = Input_train';
Input_test  = Input_test';
 
for i=1:50
   if Target_test(:,i)==[1;0;0;0;0]
        target_test(i)='a';
   end
   if Target_test(:,i)==[0;1;0;0;0]
        target_test(i)='b';
   end
   if Target_test(:,i)==[0;0;1;0;0]
        target_test(i)='c';
   end
   if Target_test(:,i)==[0;0;0;1;0]
        target_test(i)='d';
   end
   if Target_test(:,i)==[0;0;0;0;1]
        target_test(i)='e';
   end
end
SVMMd1 = fitcecoc(Input_train,target_test);
y=predict(SVMMd1,Input_test);
y=y';
target_test=target_test';
for i=1:50
   if y(i)=='a'
        Target_test(:,i)=[1;0;0;0;0];
   end
   if y(i)=='b'
        Target_test(:,i)=[0;1;0;0;0];
   end
   if y(i)=='c'
        Target_test(:,i)=[0;0;1;0;0];
   end
   if y(i)=='d';
       Target_test(:,i)=[0;0;0;1;0]; 
   end
   if y(i)=='e';
       Target_test(:,i)=[0;0;0;0;1];
   end
end  
plotconfusion(Target_test,Target_train);
