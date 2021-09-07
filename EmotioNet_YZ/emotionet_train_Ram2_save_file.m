% addpath(genpath('toolbox'))
% addpath(genpath('/eecf/cbcsl/data100/AU_det_2/Data'));

load('ShoulderPain_data.mat');
%%
AUs = [1,2,4,5,6,9,12,15,17,20,25,26];

% AUs=[4,6,7,9,10,12,20,25,26,43];

limit = 6000;

fold=5;

% division=floor(size(ShoulderPain.feature_all,1)/fold);
division = floor(size(DISFA.video_length,1)/fold);



% labels_train_intensity = ShoulderPain.GT_AU_all;
% labels_train_bin = ShoulderPain.GT_AU_all;
% labels_train_bin(labels_train_bin>0)=1;


% perm_indices = randperm(size(ShoulderPain.feature_all,1));
perm_indices = randperm(size(DISFA.video_length,1));

data_perm = [];
labels_perm = [];
intensity_perm = [];

for i=1:size(DISFA.video_length,1)
    data_perm = vertcat(data_perm,DISFA.feature_cell{perm_indices(:,i),1});
    label_temp = DISFA.GT_AU_cell{perm_indices(:,i),1};
    intensity_perm = vertcat(intensity_perm,label_temp);
    label_temp(label_temp>0)=1;
    labels_perm = vertcat(labels_perm,label_temp);
end

video_length_perm = DISFA.video_length(perm_indices,1);


% data_perm = ShoulderPain.feature_all(perm_indices,:);
% labels_perm = labels_train_bin(perm_indices,:);
% intensity_perm = labels_train_intensity(perm_indices,:);

rate_crossval=[];
F_score_crossval=[];
thresholds_crossval=[];
modelKSDA_crossval={};

% conf_cell=cell(1,60);
%%
for cross_val=1:1
    
%     test_start=(cross_val-1)*division+1;
%     test_end=cross_val*division;


    n_start=(cross_val-1)*division+1;
    n_end=cross_val*division;

    test_start = sum(video_length_perm(1:n_start)) - video_length_perm(n_start)+1;
    test_end = sum(video_length_perm(1:n_end));


    f_train_samples=data_perm;
    f_train_samples(test_start:test_end,:)=[];
    f_test_samples=data_perm(test_start:test_end,:);    
    
    labels_train_bin = labels_perm;
    labels_train_bin(test_start:test_end,:)=[];         
    labels_test=labels_perm(test_start:test_end,:);
    labels_test=labels_test+ones(size(labels_test)); 
    
    % z-score normalization
    % neutralIdx = find(sum(labels_train_bin,au) == 0);
    % neutral = mean(f_train_samples(neutralIdx,:));
    neutral = mean(f_train_samples);
    maxVec = std(f_train_samples);
    maxVec(maxVec == 0) = 1;

    tp = f_train_samples - repmat(neutral,size(f_train_samples,1),1);
    tp2 = tp./repmat(maxVec,size(f_train_samples,1),1); 
    [masterMatrix,ia,ic] = unique(tp2,'rows');
    
    labels_train = labels_train_bin(ia,:);
%     labels_train_intensity_unique = labels_train_intensity(ia,:);


    modelKSDA = cell(1,max(AUs));     
    rate_AUs=[];
    F_score=[];
    
    for au=AUs
        au

        train_label=labels_train(:,au);
        test_label = labels_test(:,au);        
       
        num_inactive = sum(train_label==0);
        num_active = sum(train_label>0);
        inactive_indices = find(train_label==0);
        active_indices = find(train_label>0);
    
        if num_inactive >= num_active
            active_train_indices = active_indices(1:min(limit,num_active));
            inactive_train_indices = inactive_indices(1:min(limit,num_active)); 
            train_label_mod = [ones(1,num_active) 2*ones(1,num_active)];
    
        else
            inactive_train_indices = inactive_indices(1:min(limit,num_inactive));
            active_train_indices = active_indices(1:min(limit,num_inactive));
            train_label_mod = [ones(1,num_inactive) 2*ones(1,num_inactive)]';
        end
    
        total_indices = [active_train_indices ;inactive_train_indices ];
        train_samples=masterMatrix([inactive_train_indices;active_train_indices],:);
        test_samples=f_test_samples;

        C = 2;
        nc=[size(active_train_indices,1),size(inactive_train_indices,1)];
    
        train_label_downsample = [zeros(size(inactive_train_indices,1),1);ones(size(active_train_indices,1),1)];

        [v, op_sigma, K1_orig, train_reordered, subClassMean, subClassLabels]=KSDA_train_Ram_Maxhomo2(train_samples',C,nc);

        
        
        pdistMatrix = pdist2(masterMatrix,train_reordered');
        [rows, col] = find(pdistMatrix == 0);
        positionVector = rows;

%         train = v'*K1_orig;
%         train_intensity_from_master = labels_train_intensity_unique(positionVector,au); %% Intensity corresponding to re_ordered data  
%         intensityMean = zeros(size(v,2),4);
%         for m2 = 1 : 4
%             posLabel = find(train_intensity_from_master == m2);
%             dataLabel = train(:,posLabel);
%             intensityMean(:,m2) = mean(dataLabel,2);
%         end

        modelKSDA{au}.v = v;
        modelKSDA{au}.numActive = num_active;
        modelKSDA{au}.op_sigma = op_sigma;
        modelKSDA{au}.K1 = K1_orig;
        modelKSDA{au}.maxVec = maxVec;
        modelKSDA{au}.neutral = neutral;
        modelKSDA{au}.labels = train_label_downsample;
        modelKSDA{au}.positionVector = positionVector;
        modelKSDA{au}.subClassMean = subClassMean;
        modelKSDA{au}.subClassLabels = subClassLabels;
%         modelKSDA{au}.intensityMean = intensityMean;
%         modelKSDA{au}.intensityLabels = train_intensity_from_master;

%         [rate , classEstimate, v ,op_sigma ]=KSDA_MaxHomoFabian(train_samples',C,nc,test_samples',test_label);
%         stats=confusionmatStats(test_label,classEstimate);
%         rate_AUs = horzcat(rate_AUs,rate);
%         F_score=horzcat(F_score,stats.Fscore(2));
% 
%         conf_cell{au}=stats.confusionMat;
    
    end

    rate_AUs=[];
    F_score=[];
    thresholds = [];
%     thresholds = [ 0.55, 0.5, 0.7, 0.7, 0.7, 0.48, 0.22, 0.6, 0.85,0.5, 0.6, 0.5 ];
%     addpath(genpath('/eecf/cbcsl/data100b/ASL_Yilin/Code/Classify_AUs/liblinear-2.1'));

    
    for au=AUs

        test_label = labels_test(:,au);        
        test_samples=f_test_samples;

%         test_label = ones(size(featuresMaster,1),1);

        [rate,classEstimate,prob] = KSDA_Testing_MasterModel_used(test_samples,test_label,modelKSDA,masterMatrix,au,'svm');
% 
%         prob(:,2) = smooth(prob(:,2));
        temp = [];
        for threshold=0.01:0.01:0.99
            estimateAU = ones(size(prob,1),1);
            estimateAU(find(prob(:,2)>=threshold)) = 2;

            stats = confusionmatStats(test_label,estimateAU);
            temp=horzcat(temp,stats.Fscore(2));
        end       
        
        max_location = find(temp==max(temp));
        thresholds(au) = 0.01*(max_location(1));

        estimateAU = ones(size(prob,1),1);
        estimateAU(find(prob(:,2)>=thresholds(au))) = 2;
        stats = confusionmatStats(test_label,estimateAU);

        rate_AUs = horzcat(rate_AUs,rate);
        F_score=horzcat(F_score,stats.Fscore(2));
        conf_cell{au}=stats.confusionMat;
        
        
    end
    
    rate_crossval = vertcat(rate_crossval,rate_AUs);
    F_score_crossval = vertcat(F_score_crossval,F_score);
    thresholds_crossval=vertcat(thresholds_crossval,thresholds);
    modelKSDA_crossval{cross_val}={modelKSDA};
    
%     save('DISFA4_cross_val_maxhomo2','rate_crossval','F_score_crossval','thresholds_crossval','-v7.3');
%     save('DISFA5_model','modelKSDA_crossval','-v7.3');
end