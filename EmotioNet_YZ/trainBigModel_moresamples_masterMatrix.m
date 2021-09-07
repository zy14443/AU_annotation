addpath(genpath('./toolbox'));
load('AUs_select.mat');
limit = 5000;
initProject;
% 
% % fold=10;
% % division=floor(size(Features,1)/fold);
% % 
% % perm_indices=randperm(size(Features,1));
% % data_perm=Features(perm_indices,:);
% % labels_perm=AU_matrix_binary(perm_indices,:);
% 
% rate_crossval=[];
% F_score_crossval=[];
% 
% %%
if ~exist('dataCell','var'),
    databaseNames = dir('*_Data.mat');
    databaseNames = {databaseNames(:).name};
    numDatabases = length(databaseNames);
    dataCell = cell(1,numDatabases);
    for k1 = 1 : numDatabases,
        tmp = load(databaseNames{k1});
        dataCell{k1} = tmp;    
    end
end
% %%
sizeDatabases = cellfun(@(x) size(x.Features),dataCell,'UniformOutput',0)';
sizeDatabases = cell2mat(sizeDatabases);
% rate = cell(numDatabases,max(select_compound_au));
% classEstimate = cell(numDatabases,max(select_compound_au));
% stats = cell(numDatabases,max(select_compound_au));
% rate_AUs = cell(numDatabases,max(select_compound_au));
% F_score = cell(numDatabases,max(select_compound_au));
% conf_cell = cell(numDatabases,max(select_compound_au));
%%

% labels_train = [];
% f_train_samples = [];
% for k2 = 1 : numDatabases,
%     dbCopy = dataCell{1,k2};
%     posActive = find(sum(dataCell{1,k2}.AU_matrix_binary,2) > 0);
%     posInactive = find(sum(dataCell{1,k2}.AU_matrix_binary,2) == 0);  
%     randVecSelect = randperm(length(posInactive));
%     if size(randVecSelect,2) ~= 0
%         labels_train = vertcat(labels_train, [dataCell{1,k2}.AU_matrix_binary(posActive,:) ;  dataCell{1,k2}.AU_matrix_binary(posInactive(randVecSelect(1:min(4000,size(posInactive,1)))),:)] );
%         f_train_samples = vertcat(f_train_samples, [dataCell{1,k2}.Features(posActive,:) ;  dataCell{1,k2}.Features(posInactive(randVecSelect(1:min(4000,size(posInactive,1)))),:)] );
%     else 
%         labels_train = vertcat(labels_train, dataCell{1,k2}.AU_matrix_binary(posActive,1:60)) ;
%         f_train_samples = vertcat(f_train_samples,dataCell{1,k2}.Features(posActive,:)) ;
%     end
% end

labels_train = [];
f_train_samples = [];
for k2 = 1 : numDatabases,
    labels_train = vertcat(labels_train, dataCell{1,k2}.AU_matrix_binary(:,1:60));
    f_train_samples = vertcat(f_train_samples,dataCell{1,k2}.Features);
end


neutralIdx = find(sum(labels_train,2) == 0);
% neutral = mean(f_train_samples(neutralIdx,:));
neutral = mean(f_train_samples);
maxVec = std(f_train_samples);
maxVec(maxVec == 0) = 1;

tp = f_train_samples - repmat(neutral,size(f_train_samples,1),1);
masterMatrix = tp./repmat(maxVec,size(f_train_samples,1),1); 

modelKSDA = cell(1,max(select_disfa_au));
for au = select_disfa_au,
    au
    train_label = labels_train(:,au);

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

    total_indices = [inactive_train_indices ;active_train_indices ];

    train_samples=f_train_samples([inactive_train_indices;active_train_indices],:);


    train_samples = train_samples - repmat(neutral,size(train_samples,1),1);

    train_samples = train_samples./repmat(maxVec,size(train_samples,1),1);
       
    C = 2;

    nc = [size(inactive_train_indices,1),size(active_train_indices,1)];
    
    train_label_downsample = [zeros(size(inactive_train_indices,1),1);ones(size(active_train_indices,1),1)];
    
    [v, op_sigma, K1_orig, train_reordered, subClassMean, subClassLabels]=KSDA_train_Ram_Maxhomo4(train_samples',C,nc);
    
    pdistMatrix = pdist2(masterMatrix,train_reordered');
    [rows, col] = find(pdistMatrix == 0);
    positionVector = rows;
    
    modelKSDA{au}.v = v;
    modelKSDA{au}.op_sigma = op_sigma;
    modelKSDA{au}.K1 = K1_orig;
    modelKSDA{au}.maxVec = maxVec;
    modelKSDA{au}.neutral = neutral;
    modelKSDA{au}.labels = train_label_downsample;
    modelKSDA{au}.positionVector = positionVector;
    modelKSDA{au}.subClassMean = subClassMean;
    modelKSDA{au}.subClassLabels = subClassLabels;
   


% % % % %     masterMatrix(rows(1,1),:) - train_reordered(:,1)';  

end
save('ModelMasterSubClassMaxhomo4.mat','modelKSDA','masterMatrix');   
%%
% for au = select_disfa_au,
%     model = modelKSDA{au};
%     save(sprintf('model_normalMoreSampRam%d',au),'model');
% end

