% addpath(genpath('toolbox'))
% addpath(genpath('/eecf/cbcsl/data100/AU_det_2/Data'));

load('DISFA_data.mat');
%%
AUs = [1,2,4,5,6,9,12,15,17,20,25,26];

limit = 6000;

fold=5;
division=floor(size(DISFA.feature_all,1)/fold);

% DISFA.GT_AU_all(DISFA.GT_AU_all>0)=1;

labels_train_intensity = DISFA.GT_AU_all;
labels_train_bin = DISFA.GT_AU_all;
labels_train_bin(labels_train_bin>0)=1;


perm_indices = randperm(size(DISFA.feature_all,1));
data_perm = DISFA.feature_all(perm_indices,:);
labels_perm = labels_train_bin(perm_indices,:);
intensity_perm = labels_train_intensity(perm_indices,:);

rate_crossval=[];
F_score_crossval=[];
conf_cell=cell(1,60);
%%
for cross_val=1:fold
    
    test_start=(cross_val-1)*division+1;
    test_end=cross_val*division;

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

        [v, op_sigma, K1_orig, train_reordered, subClassMean, subClassLabels]=KSDA_train_Ram_Maxhomo4(train_samples',C,nc);

        
        
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

%     addpath(genpath('/eecf/cbcsl/data100b/ASL_Yilin/Code/Classify_AUs/liblinear-2.1'));    

    
    for au=AUs
        au
        test_label = labels_test(:,au);        
        test_samples=f_test_samples;

%         test_label = ones(size(featuresMaster,1),1);

        [rate,classEstimate,prob] = KSDA_Testing_MasterModel(test_samples,test_label,modelKSDA,masterMatrix,au,'svm');

        temp = [];
        for threshold=0.01:0.01:0.99
            estimateAU = ones(size(prob,1),1);
            estimateAU(find(prob(:,2)>=threshold)) = 2;

            stats = confusionmatStats(test_label,estimateAU);
            temp=horzcat(temp,stats.Fscore(2));
        end       
        thresholds(au) = 0.01*(find(temp==max(temp)));

        estimateAU = ones(size(prob,1),1);
        estimateAU(find(prob(:,2)>=thresholds(au))) = 2;
        stats = confusionmatStats(test_label,estimateAU);

        rate_AUs = horzcat(rate_AUs,rate);
        F_score=horzcat(F_score,stats.Fscore(2));
        conf_cell{au}=stats.confusionMat;
        
        
    end
    
    rate_crossval = vertcat(rate_crossval,rate_AUs);
    F_score_crossval = vertcat(F_score_crossval,F_score);

    
end