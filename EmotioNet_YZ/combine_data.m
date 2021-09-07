function data_combined = combine_data(GT_Path, Classify_Path, gt_file_name, classify_file_name)


    numLandmarkFields = 134;
    numFeatureFields = 8068;
    numLabelFields = 14;

    feature_matrix_all = [];
    GT_AU_matrix_all = [];
    video_length = [];
    
    for i=1:size(gt_file_name)
        load([GT_Path,'/',gt_file_name(i).name]);
        
        fileId = fopen([Classify_Path,'/',classify_file_name(i).name],'r');
        binData = fread(fileId,'float');
        binDataMatrix = reshape(binData,[numFeatureFields,size(binData,1)/numFeatureFields])';
        fclose(fileId);

        feature_matrix = binDataMatrix(binDataMatrix(:,2)==0,:);
        GT_AU_matrix = AU_matrix(feature_matrix(:,1)+1,:);
        
        feature_matrix_all = vertcat(feature_matrix_all, feature_matrix(:,3:end));
        GT_AU_matrix_all = vertcat(GT_AU_matrix_all, GT_AU_matrix);
        video_length = vertcat(video_length, size(feature_matrix,1));
        
        
    end
% 
%     DISFA.feature_all = feature_matrix_all;
%     DISFA.GT_AU_all = GT_AU_matrix_all;
%     DISFA.video_length = video_length;

    ShoulderPain.feature_all = feature_matrix_all;
    ShoulderPain.GT_AU_all = GT_AU_matrix_all;
    ShoulderPain.video_length = video_length;

    
end