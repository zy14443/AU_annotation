load('ShoulderPain_data.mat');

% addpath(genpath('toolbox'))
% addpath(genpath('/eecf/cbcsl/data100/AU_det_2/Data'));

DISFA_AUs = [1,2,4,5,6,9,12,15,17,20,25,26];
ShoulderPain_AUs = [4,6,7,9,10,12,20,25,26];

limit = 6000;

fold=5;
division=floor(size(ShoulderPain.feature_all,1)/fold);

ShoulderPain.GT_AU_all(ShoulderPain.GT_AU_all>0)=1;


perm_indices=randperm(size(ShoulderPain.feature_all,1));
data_perm=ShoulderPain.feature_all(perm_indices,:);
labels_perm=ShoulderPain.GT_AU_all(perm_indices,:);

rate_crossval=[];
F_score_crossval=[];
conf_cell=cell(1,60);

for cross_val=1:fold
    
    test_start=(cross_val-1)*division+1;
    test_end=cross_val*division;

    f_train_samples=data_perm;
    f_train_samples(test_start:test_end,:)=[];
    f_test_samples=data_perm(test_start:test_end,:);    
      
    rate_AUs=[];
    F_score=[];
    
    for au=ShoulderPain_AUs


        train_label=labels_perm(:,au);
        train_label(test_start:test_end,:)=[];
        test_label=labels_perm(test_start:test_end,au);
        test_label=test_label+ones(size(test_label)); 

        num_inactive = sum(train_label==0);
        num_active = sum(train_label>0);
        inactive_indices = find(train_label==0);
        active_indices = find(train_label>0);
    
        if num_inactive >= num_active
            active_train_indices = active_indices;
            inactive_train_indices = inactive_indices(1:length(active_indices)); 
            train_label_mod = [ones(1,num_active) 2*ones(1,num_active)];
    
        else
            inactive_train_indices = inactive_indices;
            active_train_indices = active_indices(1:length(inactive_indices));
            train_label_mod = [ones(1,num_inactive) 2*ones(1,num_inactive)]';
        end
    
        total_indices = [active_train_indices ;inactive_train_indices ];
        train_samples=f_train_samples([active_train_indices;inactive_train_indices],:);
        test_samples=f_test_samples;

        C = 2;

        nc=[size(active_train_indices,1),size(inactive_train_indices,1)];
    
        [rate , classEstimate, v ,op_sigma ]=KSDA_MaxHomoFabian(train_samples',C,nc,test_samples',test_label);
        stats=confusionmatStats(test_label,classEstimate);
        rate_AUs = horzcat(rate_AUs,rate);
        F_score=horzcat(F_score,stats.Fscore(2));

        conf_cell{au}=stats.confusionMat;

    end

    rate_crossval = vertcat(rate_crossval,rate_AUs);
    F_score_crossval = vertcat(F_score_crossval,F_score);

    
end