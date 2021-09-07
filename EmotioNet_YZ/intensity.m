function [rate_intensity,intensityEstimate,MAE] = intensity(train_samples, test_samples, v, op_sigma, train_intensities, test_intensities, classEstimate)%, au, prior_matrix)
trainingdata=train_samples';
testingdata=test_samples';

% ---------------------- Intensities -------------------------------

% %  Setup
l=size(trainingdata,2);

A = trainingdata'*trainingdata;
dA = diag(A);
DD = repmat(dA,1,l) + repmat(dA',l,1) - 2*A;

K1=exp(-DD/(2*op_sigma^2));

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

% % %  Find distribution of each class

mean_vector=[];



for m1=1:4
    
    pos_m1=find(train_intensities==m1);
    data_m1=train(:,pos_m1);
    mean_m1=mean(data_m1',1);
    if isempty(mean_m1)
        mean_m1=1000*ones(1,size(v,1));
    end
    mean_vector=vertcat(mean_vector,mean_m1);

end

n_category=zeros(1,5);
for k1=1:5
    
    n_tp=size(find(test_intensities==(k1-1)),1);
    n_category(k1)=n_tp;
    
end

intensityEstimate=[];
test=test';
rec=0;
rec_categories=zeros(1,5);
diff_categories=zeros(1,5);
diff_just_test=zeros(1,5);
for m2=1:nXtest
    
    if classEstimate(1,m2)==1
       I=0;
    else
        
        temp = repmat(test(m2,:),4,1)-mean_vector;
        temp = temp.^2;
        dist = sum(temp',1);
        
%         prior_logs=log(prior_matrix(1:4,au));
%                  dist_with_prior=dist./prior_logs';
%         [val,I] = sort(dist_with_prior);

        [val,I] = sort(dist);

        val = val(1);
        I = I(1);
    end
    
    intensityEstimate=vertcat(intensityEstimate,I);
    
    
    if (I == test_intensities(m2)) 
        
        rec=rec+1;
        cat_tp=test_intensities(m2,1);
        rec_categories(1,cat_tp+1)=rec_categories(1,cat_tp+1)+1;
    end
    
    
    
    cat_tp=test_intensities(m2,1);
    
    if(cat_tp<5)
        diff=abs(I-test_intensities(m2));
        diff_categories(1,cat_tp+1)=diff_categories(1,cat_tp+1)+diff;
   
    end
    
    if(cat_tp<5)
        
        if(test_intensities(m2)==0)
            diff_tp=abs(I-test_intensities(m2));
        else
             diff_tp=abs(4-test_intensities(m2));
        end
        
        diff_just_test(1,cat_tp+1)=diff_just_test(1,cat_tp+1)+diff_tp;
    end
    
    
end

fractional_errors=diff_categories./n_category;
MAE=mean(fractional_errors);

fractional_just_test=diff_just_test./n_category;
MAE_just_test=mean(fractional_just_test);

rate_intensity=rec/nXtest;
% 
% figure
% hist(test_intensities);
% 
% figure
% hist(intensityEstimate);





