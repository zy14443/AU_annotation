function result = get_f1_score(classify_AU_matrix, GT_AU_matrix)

    n_frame = size(classify_AU_matrix,1);
    n_AU = size(classify_AU_matrix,2);
    
    TP = zeros(1,n_AU);
    TN = zeros(1,n_AU);
    FP = zeros(1,n_AU);
    FN = zeros(1,n_AU);
    
    
    for i=1:n_frame
        
        t1 = (classify_AU_matrix(i,:)>0);
        t2 = (GT_AU_matrix(i,:)>0);
        
        TP = TP+and(t1,t2);
        
        TN = TN+(~or(t1,t2));
        
        FP = FP+(t1>t2);
        
        FN = FN+(t1<t2);        
        
    end
    
    result.TP = TP;
    result.TN = TN;
    result.FP = FP;
    result.FN = FN;

end