clear all;
myVoice = audiorecorder;
figure();
myVoice.StartFcn = 'disp(''Start speaking.'')';
myVoice.StopFcn = 'disp(''End of recording.'')';
disp('enter the number of signals')
signals=input('signals');
%data=zeros(7,signals);
for i=1:signals
opt=input('press a key to start');
record(myVoice, 2);
pause(3);
b=getaudiodata(myVoice);
rollno=input('enter roll no');
roll=num2str(rollno);
plot(b);
name=strcat('neutral',roll,'.wav');
audiowrite(name,b,8000);                                                                                            
end