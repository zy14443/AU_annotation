function [v, op_sigma, K1, trainingdata, subClassMean, subClassLabels]=KSDA_train_Ram_MaxhomoCrazy(trainingdata,C,nc)



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

numSubClass=4;
Sigma=zeros(1,numSubClass);
fval=zeros(1,numSubClass);
for ii=2:numSubClass,
    H = ii*ones(1,C);
    NH = get_NH(C,H,nc);
    X0=sqrt(mean_DD/2);
    [A,label,sub_label] = calcA(H,NH,l,C);
    [Sigma(ii),fval(ii)] = fminunc(@(sigma)MaxhomoCrazy(H,label,sub_label, C, A, sigma,DD),X0,options);
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

train = v' * K1;

subClassMean = zeros(size(v,2),size(NH,2));
ctr = 1;
for m1 = 1 : size(NH,2)
%     
    subClassMean(:,m1)  = mean(train(:,ctr:ctr + NH(m1) - 1),2);
    ctr = ctr + NH(m1);
end
subClassLabels = [zeros(1,op_H),ones(1,op_H)];


% intensityMean = zeros(size(v,2),4);
% for m2 = 1 : 4
%     posLabel = find(intensity == m2);
%     dataLabel = train(:,posLabel);
%     intensityMean(:,m2) = mean(dataLabel,2);
% end



    





