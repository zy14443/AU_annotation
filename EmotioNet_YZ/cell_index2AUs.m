function output = cell_index2AUs(input, AUs)

    output = cell(1,60);
    
    k=1;
    for i=AUs
        
        output{i} = input{k};
        k=k+1;
        
        
    end


end