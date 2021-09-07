
function [rate classEstimate v op_sigma ]=KSDA_MaxHomoFabian(trainingdata,C,nc,testingdata,test_label)
% 
% trainingdata=train_samples';
% testingdata=test_samples';


%------------------------------- training stage -------------------------

%%% Nearest Neighbor clustering of the data

Ytrain = NNclassclustering2(trainingdata',C,nc);
trainingdata=Ytrain';
l=size(trainingdata,2);

 %%% get pairwise distance matrix

A = trainingdata'*trainingdata;
dA = diag(A);
DD = repmat(dA,1,l) + repmat(dA',l,1) - 2*A;

s1=sum(sum(DD,1));
num=l*(l-1)/2;
mean_DD=s1/2/num;
options = optimset('LargeScale','off', 'Display','iter', 'GradObj','off',...
    'HessUpdate','bfgs', 'TolX',1e-10, 'MaxFunEvals',5000, 'MaxIter',10000);

numSubClass=5;
Sigma=zeros(1,numSubClass);
fval=zeros(1,numSubClass);
for ii=1:numSubClass,
    H = ii*ones(1,C);
    NH = get_NH(C,H,nc);
    X0=sqrt(mean_DD/2);
    [A,label,sub_label] = calcA(H,NH,l,C);
    [Sigma(ii),fval(ii)] = fminunc(@(sigma)Maxhomo2(H,label,sub_label, C, A, sigma,DD),X0,options);
end
[F,ind]=min(fval);
op_H=ind;
op_sigma=Sigma(ind);

%%% perform KSDA after selecting optimal parameters

H = op_H*ones(1,C);
NH = get_NH(C,H,nc);
K1=exp(-DD/(2*op_sigma^2));
[A,label,sub_label] = calcA(H,NH,l,C);

display('start KSDA');

v=KSDA2(C,trainingdata,H,NH,K1,A);

% ---------------------- testing stage -------------------------------


display('start testing');

 train=v'*K1;
 nXtest=size(testingdata,2);
 dd=zeros(nXtest,size(trainingdata,2));
for i=1:nXtest
    B=trainingdata-repmat(testingdata(:,i),1,l);
    B=B.^2;
    dd(i,:)=sum(B,1);
end
dd=dd';
K2=exp(-dd/(2*op_sigma^2));

test=v'*K2;

%%% nearest neighbor classifier
      
[rate classEstimate ]=NearestNeighbor(train',test',test_label,C,nc);



end