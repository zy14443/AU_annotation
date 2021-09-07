clear
addpath(genpath('./toolbox'));
load('AUs_select.mat');
limit = 6000;
initProject;

select_combined_au = [1,2,4,5,6,7,9,10,12,15,17,18,20,24,25,26];

if ~exist('dataCell','var'),
    databaseNames = dir('*_Data.mat')
    databaseNames = {databaseNames(:).name};
    numDatabases = length(databaseNames);
    dataCell = cell(1,numDatabases);
    for k1 = 1 : numDatabases,
        tmp = load(databaseNames{k1});
        dataCell{k1} = tmp;    
    end
end

sizeDatabases = cellfun(@(x) size(x.Features),dataCell,'UniformOutput',0)';
sizeDatabases = cell2mat(sizeDatabases);

labels_train_orig = [];
f_train_samples = [];
labels_train_intensity = [];
for k2 = 1 : numDatabases,
    labels_train_orig = vertcat(labels_train_orig, dataCell{1,k2}.AU_matrix_binary(:,1:60));
    f_train_samples = vertcat(f_train_samples,dataCell{1,k2}.Features);
    labels_train_intensity = vertcat(labels_train_intensity, dataCell{1,k2}.AU_matrix_all(:,1:60));
end

% %  Some of the intensities are >1 when binary label is 0 - reseting the
% binary labels back to orig values
labels_train_intensity = floor(labels_train_intensity);
labels_train_orig(find(labels_train_intensity>0&labels_train_intensity<6)) = 1;


neutralIdx = find(sum(labels_train_orig,2) == 0);
% neutral = mean(f_train_samples(neutralIdx,:));
neutral = mean(f_train_samples);
maxVec = std(f_train_samples);
maxVec(maxVec == 0) = 1;

tp = f_train_samples - repmat(neutral,size(f_train_samples,1),1);
tp2 = tp./repmat(maxVec,size(f_train_samples,1),1); 
[masterMatrix,ia,ic] = unique(tp2,'rows');
labels_train = labels_train_orig(ia,:);
labels_train_intensity_unique = labels_train_intensity(ia,:);


modelKSDA = cell(1,max(select_disfa_au));
for au = 2%select_combined_au,
    au
    train_label = labels_train(:,au);

    num_inactive = sum(train_label==0);
    num_active = sum(train_label>0&train_label<6);
    inactive_indices = find(train_label==0);
    active_indices = find(train_label>0&train_label<6);

    if num_inactive >= num_active
        posPerm = randperm(num_active);
        active_train_indices = active_indices(posPerm(1:min(limit,num_active)));
        inactive_train_indices = inactive_indices(posPerm(1:min(limit,num_active))); 
        train_label_mod = [ones(1,min(limit,num_active)) 2*ones(1,min(limit,num_active))];

    else
        posPerm = randperm(num_inactive);
        inactive_train_indices = inactive_indices(posPerm(1:min(limit,num_inactive)));
        active_train_indices = active_indices(posPerm(1:min(limit,num_inactive)));
        train_label_mod = [ones(1,num_inactive) 2*ones(1,num_inactive)]';
    end

    total_indices = [inactive_train_indices ;active_train_indices ];

    train_samples=masterMatrix([inactive_train_indices;active_train_indices],:);
       
    C = 2;
    nc = [size(inactive_train_indices,1),size(active_train_indices,1)];
    
    train_label_downsample = [zeros(size(inactive_train_indices,1),1);ones(size(active_train_indices,1),1)];
    
    [v, op_sigma, K1_orig, train_reordered, subClassMean, subClassLabels]=KSDA_train_Ram_MaxhomoCrazy(train_samples',C,nc);
    
    pdistMatrix = pdist2(masterMatrix,train_reordered');
    [rows, col] = find(pdistMatrix == 0);
    positionVector = rows;
       
    train = v'*K1_orig;
    train_intensity_from_master = labels_train_intensity_unique(positionVector,au); %% Intensity corresponding to re_ordered data  
    intensityMean = zeros(size(v,2),4);
    for m2 = 1 : 4
        posLabel = find(train_intensity_from_master == m2);
        dataLabel = train(:,posLabel);
        intensityMean(:,m2) = mean(dataLabel,2);
    end
    
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
    modelKSDA{au}.intensityMean = intensityMean;
    modelKSDA{au}.intensityLabels = train_intensity_from_master;
    

%     save('ModelMasterCrazyIntensity.mat','modelKSDA','-v7.3');  
end

% save('MasterMatrixCrazyIntensity.mat','masterMatrix','-v7.3');

%%
% % 
% % for au = 25%select_combined_au,
% %     model = modelKSDA{au};
% %     v = model.v;
% %     positionVector = model.positionVector;
% %     train_intensity_from_master = labels_train_intensity_unique(positionVector,au);   
% %     K1 = model.K1;
% %     
% %     train = v'*K1;
% %     
% %     intensityMean = zeros(size(v,2),4);
% %     for m2 = 1 : 4
% %         posLabel = find(train_intensity_from_master == m2);
% %         dataLabel = train(:,posLabel);
% %         intensityMean(:,m2) = mean(dataLabel,2);
% %     end
% %        
% % end


% % for au = select_disfa_au,
% %     model = modelKSDA{au};
% %     save(sprintf('model_normalMoreSampRam%d',au),'model');
% % end

