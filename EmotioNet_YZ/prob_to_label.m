function result = prob_to_label(AU_prob_matrix,thresholds)

    n_frame = size(AU_prob_matrix,1);
    n_AU = size(AU_prob_matrix,2);

    smooth_prob_matrix = zeros(size(AU_prob_matrix));
    for j=1:n_AU        
        smooth_prob_matrix(:,j) = smooth(AU_prob_matrix(:,j),3);        

%         fc = 2;
%         fs = 20;
%         [b,a] = butter(10,fc/(fs/2));
%         freqz(b,a)

%         smooth_prob_matrix(:,j) = filtfilt(b,a,AU_prob_matrix(:,j)); 
    end
          
    result = zeros(size(AU_prob_matrix));
    
    for i=1:n_frame
        
        temp = zeros(1,n_AU);
        temp(smooth_prob_matrix(i,:)>thresholds)=1;
        
        result(i,:)=temp;
        
    end
   
end