load('DISFA_data.mat');

n_video = size(DISFA.video_length,1);

DISFA.feature_cell = cell(n_video,1);
DISFA.GT_AU_cell = cell(n_video,1);


for i=1:n_video
    
    n_start = sum(DISFA.video_length(1:i,:))-DISFA.video_length(i,:)+1;
    n_end = sum(DISFA.video_length(1:i,:));
    
    DISFA.feature_cell{i,1} = DISFA.feature_all(n_start:n_end,:);
    DISFA.GT_AU_cell{i,1} = DISFA.GT_AU_all(n_start:n_end,:);
    
end