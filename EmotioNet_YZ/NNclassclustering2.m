function sortedtrain = NNclassclustering2(trainingdata, C, nc)
% first step: find two samples (element 1 and element 2) in each class
% which have the largest distance between each other
%
% second step: sort the data such that: element 1 and element 2 are the
% 1st and nth sample in the sorted training data, and 1~n/2 samples are near
% element 1 and n/2+1~n samples are near element2 

% input: trainingdata: n-by-p matrix, all the data
%        C: number of classes
%        nc: c-by-1 matrix containing the number of samples for each class
% output:
%        trainingdata: the sorted training data


[n,p]=size(trainingdata);

 
element1=zeros(C,1);   %record the index of the two most distant samples in each class
element2=zeros(C,1);
dist=0;      %record the distance
larg_dist=zeros(1,C);%record the largest distance of every class 10
start=0;
DD=cell(1,C);
for k=1:C
    %k    
    X = trainingdata(start+1:start+nc(k),:);
    numSamp=size(X,1);
    A = X*X';
    dA = diag(A);
    DD{k} = triu(repmat(dA,1,numSamp) + repmat(dA',numSamp,1) - 2*A);
    [larg_dist(k),indx]=max(DD{k}(:));
    [row,col]=ind2sub(size(DD{k}),indx);
    element1(k)=row+start;
    element2(k)=col+start;
    start=sum(nc(1:k));
end


%sorted data
start=0;
sortedtrain=zeros(n,p);
for k=1:C
    %k
    
    key1=trainingdata(element1(k),:);   % in class K, the key element 
    key2=trainingdata(element2(k),:);    
    
    sortedtrain(start+1,:)= key1; % the first and last elements in sorted trainindata are the keys
    sortedtrain(start+nc(k),:) = key2;
    
    
    eleOld1=element1(k)-start;
    eleOld2=element2(k)-start;
%     DDtemp=DD{k};    
    numSamp = size(DD{k},2);
    DDtemp=DD{k}+DD{k}';
    DDtemp(1:numSamp+1:end)=inf;
    for k1=1:fix((nc(k)-2)/2),
        DDtemp(eleOld1,eleOld2)=inf;
        DDtemp(eleOld2,eleOld1)=inf;                
        [~,eleOld1]=min(DDtemp(element1(k)-start,:));
        DDtemp(:,eleOld1)=inf;
        [~,eleOld2]=min(DDtemp(element2(k)-start,:));
        DDtemp(:,eleOld2)=inf;
        key1=trainingdata(eleOld1+start,:);   % in class K, the key element 
        key2=trainingdata(eleOld2+start,:);    
   
        sortedtrain(start+1+k1,:)= key1; % the first and last elements in sorted trainindata are the keys
        sortedtrain(start+nc(k)-k1,:) = key2;                               
    end
    if (mod(nc(k),2)~=0)
        DDtemp(eleOld1,eleOld2)=inf;
        DDtemp(eleOld2,eleOld1)=inf;        
        DDtemp(:,eleOld2)=inf;
        DDtemp(:,eleOld1)=inf;
        [~,eleOld1]=min(DDtemp(element1(k)-start,:));
        sortedtrain(start+fix(nc(k)/2)+1,:)=trainingdata(eleOld1+start,:);
    end

    start=sum(nc(1:k));
end

% [larg_dist(k),indx]=max(DDtemp(:));
%         [eleNew1,eleNew2]=ind2sub(size(DDtemp),indx);
%         if DD{k}(eleNew1,eleOld1)+DD{k}(eleOld1,eleNew1)<DD{k}(eleNew2,eleOld1)+DD{k}(eleOld1,eleNew2),
%             eleOld1=eleNew1;
%             eleOld2=eleNew2;
%         else
%             eleOld1=eleNew2;
%             eleOld2=eleNew1;
%         end
%         key1=trainingdata(eleOld1+start,:);   % in class K, the key element 
%         key2=trainingdata(eleOld2+start,:);    
%     
%         sortedtrain(start+1+k1,:)= key1; % the first and last elements in sorted trainindata are the keys
%         sortedtrain(start+nc(k)-k1,:) = key2;       
