function l3b
% %% feature extraction block
% % This block of the code extracts features from the audio files.
% % Specifically we are looking to extract mfcc features. the accompanying 
% % mfcc folder has the necessary files to extract the features. The data is
% % stored as genre.mat at the end of feature extraction process. Each row in
% % genre.mat corresponds to the mfcc's obtained from one audio file. Each
% % column refers to one of the mfcc. The last column in the genre label.
% % DO NOT play around with this part of the code. We will use these features
% % for the first part of the lab.
% % After genre.mat has been created, you would not need to run this block of
% % code again.
% 
% % add path containing the mfcc files
% addpath ./mfcc/mfcc;
% addpath ./data;
% % folder containing the data
% foldername ='./data';
% % musical genres for classification
% genres = {'classical', 'hiphop', 'metal', 'pop'};
% data = [];
% 
% % Define variables for extracting MFCCs
% Tw = 500;              % analysis frame duration (ms)
% Ts = 100;               % analysis frame shift (ms)
% alpha = 0.97;           % preemphasis coefficient
% M = 20;                 % number of filterbank channels
% C = 12;                 % number of cepstral coefficients
% L = 22;                 % cepstral sine lifter parameter
% LF = 300;               % lower frequency limit (Hz)
% HF = 3700;              % upper frequency limit (Hz)
% 
% % loop through all the files for extracting the MFCC
% 
% for i=1:size(genres,2)
%     str=sprintf('%s/%s*',foldername, genres{i});
%     genrelist = dir(str);
%     for j=1:size(genrelist,1);
%         % read the audio file
%         [Y FS] = audioread(genrelist(j).name);
%         % extract the MFCC
%         [MFCCs, FBEs, frames] = ...
%             mfcc( Y, FS, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );
%         % append the MFCCs and the genre label
%         data = [data; MFCCs(:)' i];
%     end
% end
% save('genre', 'data');
% 
% 
% 
% %% the multilayer perceptron block
% % load the genre.mat dataset and train a multilayer perceptron that uses
% % sigmoid function as the activation function for hidden and output layer
% % units.
clear all;

A=importdata('genre.mat',' ',1);

% disp(A(:,1:200)); 

            
            
%             [pp,qq]=size(B);
%             A=[];
%             for ii=1:qq
%              
%                if (rem(ii,13)~=1)
%                   A=[A B(:,ii)] ;
%                else
%                    ttt = -20+(40)*rand(pp,1);
%                    A=[A ttt];
%                end
%                
%             end
            

            








[n,m]=size(A);

TEMP1=zeros(n,4);
for i=1:n
   if (A(i,m)==1)
       TEMP1(i,1)=1;
   elseif (A(i,m)==2)
       TEMP1(i,2)=1;
   elseif (A(i,m)==3)
       TEMP1(i,3)=1;
   else
       TEMP1(i,4)=1;
   end    
end
%disp('Rehan');
%disp(size(A));

A(:,m)=[];
%disp(size(A));
A=[A TEMP1];
%disp(size(A));
%disp(TEMP1);
[n,m]=size(A);
kfold=5;
C1=[];
C2=[];
C3=[];
C4=[];

x=41;

for m1=1:x    
    C1(m1,:) = A(m1,:);
end

m1=x+1;

for m2 = 1:x+1
   
    C2(m2,:) = A(m1,:);
    m1=m1+1;
end


for m3  = 1:x
    
   C3(m3,:) =A(m1,:);
   m1=m1+1;
end

%disp(m1);

for m4=1:x

    C4(m4,:) = A(m1,:);
    m1=m1+1;
end

%disp(size(C1));
%disp(size(C2));
%disp(size(C3));
%disp(size(C4));

val=floor(n/kfold);
val1=floor(val/4);


C=[];
for l=1:kfold
    B=[];
    j=1;
    if (l<kfold)
       for i=1:val1
           B(j,:)=C1(1,:);
           
           B(j+1,:)=C2(1,:);

           B(j+2,:)=C3(1,:);
           B(j+3,:)=C4(1,:);
           C1(1,:)=[];
           C2(1,:)=[];
           C3(1,:)=[];
           C4(1,:)=[];
           j=j+4;
           
       end
       C=[C;B];
    else
        B=[C1;C2;C3;C4];
        [p,q]=size(B);
        D=[];
        iporder = randperm(p);
        for d=1:p
            D(d,:)=B(iporder(d),:);
        end
        C=[C;D];
    end
    
end
avgfintesterror=[];


hArray=[4 16 32 64 128 256 300];
etaArray=[0.005 0.008 0.01 0.05];
    
for kk=1:7
    
    H=hArray(1,kk);
  avgtesterror=[];
   a=0.003;
   b=0.009;
   c=0.003;
    for e=1:4
        
        eta=etaArray(1,e);
        nEpochs=1000;
        testerror=[];
       
        
        for i=1:kfold
            stt=sprintf('\nH = %d     learning rate = %f     fold number = %d',H,eta,i);
            disp(stt);
            
            X=C;
            Xtest = [];
            
            if(i<kfold)
            
                Xtest = X(((i-1)*val1*4+1):i*val1*4,:);
                X(((i-1)*val1*4+1):i*val1*4,:)=[];
                
            else
                Xtest = X(((i-1)*val1*4+1):n,:);
                X(((i-1)*val1*4+1):n,:)=[];
            end
            Ytest=Xtest(:,(m-3):m);
            Xtest(:,(m-3):m)=[];
            Y=X(:,(m-3):m);
            X(:,(m-3):m)=[];
            
            [w v] = mlptrain(X,Xtest,Ytest, Y, H, eta, nEpochs);

            
            
            
            
            
            Mean=mean(X);
Max=max(X);
Min=min(X);
M=size(Xtest,1);

Mean1=ones(M,1)*Mean;
Dev1=ones(M,1)*(Max-Min);


Xtest=Xtest-Mean1;
Xtest=Xtest ./ (Dev1);
            
            %ydash
            ydash = mlptest(Xtest, w, v);
           
            
            
            
            correct1=0;
             for iii=1:M
               if(ydash(iii,:)==Ytest(iii,:))
                   correct1=correct1+1;
               end
             end
             
             accuracy1 = correct1/M;
             error=1-accuracy1;

                
            testerror =[testerror error];
            
            
        end
      mean1=mean(testerror,2);
      avgtesterror=[avgtesterror mean1];
        
    end
    str=sprintf('\nError for H = %d and for different learning rate values values',H);
    disp(str);
    disp(avgtesterror);
    
%     [mini,r]=min(avgtesterror');
%     col=[mini(1,1);(a+c*(r(1,1)-1))];
%     avgfintesterror=[avgfintesterror col];
%     
    
end

%disp(avgfintesterror);

function [w v] = mlptrain(X,Xtest,Ytest, Y, H, eta, nEpochs)

% X - training data of size NxD
% Y - training labels of size NxK
% H - the number of hiffe
% eta - the learning rate
% nEpochs - the number of training epochs
% define and initialize the neural network parameters

% number of training data points
N = size(X,1);
% number of inputs
D = size(X,2); % excluding the bias term
% number of outputs
K = size(Y,2);
M= size(Xtest,1);
%disp(X);
%disp(Y);
%disp(K);

% weights for the connections between input and hidden layer
% random values from the interval [-0.3 0.3]
% w is a Hx(D+1) matrix
alp=0.4;
w = -0.3+(0.6)*rand(H,(D+1));
v = -0.3+(0.6)*rand(K,(H+1));
w1= zeros(H,(D+1));
v1= zeros(K,(H+1));



Mean=mean(X);
Max=max(X);
Min=min(X);

% Mean2=mean(Xtest);
% Max2=max(Xtest);
% Min2=min(Xtest);
Mean1=ones(M,1)*Mean;
Dev1=ones(M,1)*(Max-Min);

Mean=ones(N,1)*Mean;
Dev=ones(N,1)*(Max-Min);

X=X-Mean;
X=X ./ (Dev);


Xtest=Xtest-Mean1;
Xtest=Xtest ./ (Dev1);
%W=dlmread('text1.txt');
%V=dlmread('text2.txt');
%w=zeros(H,(D+1));
%v=zeros(K,(H+1));
%for creatW=1:D+1
 %  w(:,creatW) = W(1,creatW)-W(2,creatW)+2*W(2,creatW)*rand(H,1);
%end

%for creatV=1:H+1
 %  v(:,creatV) = V(1,creatV)-V(2,creatV)+2*V(2,creatV)*rand(K,1);
%end










%disp(size(X));
% randomize the order in which the input data points are presented to the
% MLP

% mlp training through stochastic gradient descent


batchSize=10;
mea=0;
mea1=0;
ydash1=zeros(N,4);
for epoch = 1:nEpochs
    SumV=zeros(K,H+1);
    SumW=zeros(H,D+1);
    iporder = randperm(N);
    for n = 1:N
        % the current training point is X(iporder(n), :)
        current = X(iporder(n),:);
        current = [1 current];
        
        currentY = Y(iporder(n),:);
         % forward pass
        % --------------
        % input to hidden layer
        % calculate the output of the hidden layer units - z
        % ---------
        %'TO DO'%
        
        
        
        
        
%         hiddenOutput=zeros(1,H);
%         
%         for j=1:H
%             
%             w_h = w(j,:);
%             
%             op =  w_h*current';
%             
%             
%             hiddenOutput(1,j) =sigmf(op,[1 0]);
%            
%         end 
%         hiddenOut=[1 hiddenOutput];

             hiddenOutput=(w*(current'))';
             hiddenOutput=sigmf(hiddenOutput,[1,0]);
             
            hiddenOut=[1 hiddenOutput]; 
            
            
        
        % ---------
        % hidden to output layer
        % calculate the output of the output layer units - ydash
        % ---------
        %'TO DO'%
        
        
         
        
%         Output=zeros(1,K);
%         sum=0;
%           for j=1:K
%             
%             v_h = v(j,:);
%             
%             op1 =  exp(v_h*hiddenOut');
%             Output(1,j) =op1;
%             sum=sum+op1;
%            
%           end  
%           Output=Output/sum;





         Output=(v*(hiddenOut'))';
         sum=0;
         for j=1:K
             Output(1,j)=exp(Output(1,j));
             sum=sum+Output(1,j);
         end
         Output=Output/sum;
         
         
         
         
         
         
        
        % ---------
        
        % backward pass
        % ---------------
        % update the weights for the connections between hidden and
        % outlayer units
        % ---------
        %'TO DO'%
        % ---------
        
        
%         
%         for k=1:K
%             
%             
%             for h=1:H+1
%                 
%                 delV = eta* hiddenOut(1,h)*(Output(1,k)-currentY(1,k))+alp*v1(k,h);
%                 v1(k,h)=delV;
%                 v(k,h) = v(k,h)-delV;
%                 
%        
%                 
%             end    
%             
%         end   



        delV=eta*((Output-currentY)')*hiddenOut; %+ alp*v1;
        
        %v1=delV;
          SumV=SumV+delV;
        
        if(rem(n,batchSize)==0 || n==N)
            if(rem(n,batchSize)==0)
                ttt1 = SumV/batchSize;
            else
                ttt1=SumV/(rem(n,batchSize));
            end
            v=v-ttt1;
            SumV=zeros(K,H+1);
          
        end
       
   
        
        
        
%         for h=1:H
%             
%            for j=1:D+1
%              
%                delW =0;
%                for k=1:K
%                    hiddenOutput(1,h)= hiddenOutput(1,h)/100;
%                    delW=delW+(Output(1,k)-currentY(1,k))*v(k,h+1)*hiddenOutput(1,h)*(1-hiddenOutput(1,h))*current(1,j);
%                    
%                end
%                delW=eta*delW+alp*w1(h,j);
%                w1(h,j)=delW;
%                w(h,j)=w(h,j)-delW;
%                
%            end
%             
%         end
           
        
           temp11=ones(H,1);
           for p=1:H
               temp11(p,1)=(hiddenOutput(1,p))*(1-(hiddenOutput(1,p)));
           end
           
           delW=(temp11*current);
           temp22= (Output-currentY)*v(:,2:H+1) ;
           for p=1:H
           
               delW(p,:)=eta*temp22(1,p)*delW(p,:);
           
           end
           
        SumW=SumW+delW;
        if(rem(n,batchSize)==0 || n==N)
            if(rem(n,batchSize)==0)
                ttt2 = SumW/batchSize;
            else
                ttt2=SumW/(rem(n,batchSize));
            end
            w=w-ttt2;
            SumW=zeros(H,D+1);
        
        end
          
    end
    
    
    
    ydash = mlptest(X, w, v);
    ydash1= mlptest(Xtest, w, v);
    % compute the training error
    % ---------
     correct=0;
     for i=1:N
       if(ydash(i,:)==Y(i,:))
           correct=correct+1;
       else
           %disp('Different');
       end
     end
     correct;
     accuracy = correct/N;
     error=1-accuracy;
     
     %disp(error);
     
     
     correct1=0;
     for i=1:M
       if(ydash1(i,:)==Ytest(i,:))
           correct1=correct1+1;
       else
           %disp('Different');
       end
     end
     correct1;
     accuracy1 = correct1/M;
     error1=1-accuracy1;
     %disp(error);
     
     str=sprintf('Epoch = %d     CorrectTrain = %d     Train error = %f     CorrectTest = %d     Test error = %f',epoch,correct,error,correct1,error1);
     disp(str);
     
     
%      
%      correct1=0;
%      temp=zeros(1,N);
%      for i=1:N
%          if(ydash(i,:)==ydash1(i,:))
%            correct1=correct1+1;
%          end
%      end
%      
%      if (correct1==N)
%         disp('Same'); 
%      else
%          disp('Not same');
%      end
%      ydash1=ydash;
%      
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
%     
%     %'TO DO'% uncomment the next line after adding the necessary code
%     trainerror =[trainerror error];
%     % ---------
    %disp(sprintf('training error after epoch %d: %f\n',epoch, trainerror(epoch)));
end

return;

function ydash = mlptest(X, w, v)
% forward pass of the network

% number of inputs
N = size(X,1);
H=size(w,1);
% number of outputs
K = size(v,1);
ydash=zeros(N,4);
pp=ones(N,1);
XX=[pp X];
   for n = 1:N
        % the current training point is X(iporder(n), :)
        current = XX(n,:);
        
        hiddenOutput=zeros(1,H);
        for j=1:H
            
            w_h = w(j,:);
            
            op =  w_h*current';
            hiddenOutput(1,j) =sigmf(op,[1 0]);
           
        end    
            hiddenOut=[1 hiddenOutput];  
            
        
       
        output=zeros(1,K);
        sum=0;
          for j=1:K
            
            v_h = v(j,:);
            
            op1 =  exp(v_h*hiddenOut');
            output(1,j) =op1;
            sum=sum+op1;
           
          end  
        output=output/sum;
        ydash(n,:)=output;
   end
   
   
  % disp(ydash);
  
   for i=1:N
       max=-1;
       temp=-1;
      for j=1:K
          if (max<=ydash(i,j))
              max=ydash(i,j);
              temp=j;
          end
      end
      for ppp=1:K
         if (ppp==temp)
            ydash(i,ppp)=1; 
         else
            ydash(i,ppp)=0;    
         end
          
      end
       
   end
% forward pass to estimate the outputs
% --------------------------------------
% input to hidden for all the data points
% calculate the output of the hidden layer units
% ---------
%'TO DO'%
% ---------% hidden to output for all the data points
% calculate the output of the output layer units
% ---------
%'TO DO'%
% ---------
return;



% ** Your Code **