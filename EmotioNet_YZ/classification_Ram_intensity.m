clear

load('Emotion_Data.mat');
load('AUs_select.mat');
perm_indices=randperm(size(Features,1));

train_range=[1:2500];
test_range=[3500:4000];

data_perm=Features(perm_indices,:);

f_train_samples=data_perm(train_range,:);
f_test_samples=data_perm(test_range,:);
rate_AUs=[];
rate_intensity=[];
F_score=[];
MAE_vector=[];

for au=1:1%select_disfa_au
    
    labels_perm=AU_matrix_binary(perm_indices,:);
    train_label=labels_perm(train_range,au);
    test_label=labels_perm(test_range,au);
    test_label=test_label+ones(size(test_label)); 
    pos_fractions=find(test_label>2);
    test_label(pos_fractions,1)=2;

    intensities_perm=AU_matrix_all(perm_indices,:);
    train_intensities_orig=floor(intensities_perm(train_range,au));
    test_intensities=floor(intensities_perm(test_range,au));
    
    num_inactive = sum(train_label==0);
    num_active = sum(train_label>0);
    inactive_indices = find(train_label==0);
    active_indices = find(train_label>0);

    if num_inactive >= num_active
        active_train_indices = active_indices;
        inactive_train_indices = inactive_indices(1:length(active_indices)); %(end-length(active_indices):end);%
        train_label_mod = [ones(1,num_active) 2*ones(1,num_active)];

    else
        inactive_train_indices = inactive_indices;
        active_train_indices = active_indices(1:length(inactive_indices)); %(end-length(inactive_indices):end);%
        train_label_mod = [ones(1,num_inactive) 2*ones(1,num_inactive)]';
    end

    total_indices = [active_train_indices ;inactive_train_indices ];
    train_intensities=train_intensities_orig(total_indices,1);
    train_samples=f_train_samples([active_train_indices;inactive_train_indices],:);
    test_samples=f_test_samples;

    train_samples = train_samples-repmat(mean(neutral),size(train_samples,1),1);
    test_samples =  test_samples-repmat(mean(neutral),size(test_samples,1),1);

    C = 2;
    % % % % % % nc =[num_inactive num_active];
    nc=[size(active_train_indices,1),size(inactive_train_indices,1)];

    [rate , classEstimate, v ,op_sigma ]=KSDA_MaxHomoFabian(train_samples',C,nc,test_samples',test_label);
    stats=confusionmatStats(test_label,classEstimate);
    rate_AUs = horzcat(rate_AUs,rate);
    F_score=horzcat(F_score,stats.Fscore(2));
% %     
    [rate_int,intensityEstimate,MAE]=intensity(train_samples, test_samples, v, op_sigma, train_intensities, test_intensities, classEstimate);
    MAE_vector=vertcat(MAE_vector,MAE);
    rate_intensity=horzcat(rate_intensity,rate_int);
%     
end





