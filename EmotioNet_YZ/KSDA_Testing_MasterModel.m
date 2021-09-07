function [rate,classEstimate,probClass] = KSDA_Testing_MasterModel(testingdata,test_label,model,masterMatrix,au,method)

if nargin<6,
    method = 'svm';
end


trainingdata = masterMatrix(model{1,au}.positionVector,:)';
op_sigma = model{1,au}.op_sigma;
nc = [nnz(model{1,au}.labels),nnz(model{1,au}.labels)];
% svmModel = model.svmModel;
neutral = model{1,au}.neutral;
maxVec = model{1,au}.maxVec;
v= model{1,au}.v;
trainMat = v'*model{1,au}.K1;

testingdata = testingdata - repmat(neutral,size(testingdata,1),1);
testingdata = testingdata./repmat(maxVec,size(testingdata,1),1);
testingdata = testingdata';


dd = bsxfun(@plus,full(dot(trainingdata,trainingdata,1)),full(dot(testingdata,testingdata,1))')-full(2*(testingdata'*trainingdata));
dd = dd';
K2 = exp(-dd/(2*op_sigma^2));
test = v'*K2;


trainingLabels = [];
classOut = zeros(length(nc),length(test_label));

if strcmpi(method,'mahal'),
    disp('0');
    startClass = 1;
    for k1 = 1 : length(nc),    
        trainingLabels = [trainingLabels k1*ones(1,length(startClass:startClass+nc(k1)-1))];  
        classOut(k1,:) = mahal(test',trainMat(:,startClass:startClass+nc(k1)-1)');    
        startClass = startClass + nc(k1);
    end
    [~,classEstimate] = min(classOut);
    probClass = 1 - classOut./repmat(sum(classOut,1),size(classOut,1),1);
    probClass = probClass(1,:);
    probClass = probClass';
    rate = mean(classEstimate == test_label);
elseif strcmpi(method,'svm')    
    startClass = 1;
    for k1 = 1 : length(nc),    
        trainingLabels = [trainingLabels k1*ones(1,length(startClass:startClass+nc(k1)-1))];         
        startClass = startClass + nc(k1);
    end
    svmModel = train(trainingLabels', sparse(trainMat'),'-c 10 -s 7 -q');
    [classEstimate, accuracy_results, probClass] = predict(test_label, sparse(test'), svmModel,'-b 1 -q');
%     probClass(:,2) = [];
    rate = accuracy_results(1);    
end
