function J=Maxhomo2(H,label,sub_label, C, A, sigma,DD)

% Homoscedastic criterion

% Kernel Optimization in Discriminant Analysis. IEEE Transactions on
% Pattern Analysis and Machine Intelligence. 

% This is an implementation of the Homoscedastic criterion. 

% Input: 

% C: number of classes

% H: a 1-by-C vector with each element indicating the number of subclassses
% in each class. 

% nh: a 1-by-C*sum(H) vector with each element indicating the number of
% samples in each subclass.

% l: number of the training samples

% sigma: the RBF kernel parameter

% DD: the Euclidean distance matrix of the pairwise samples


% Output:

% J: the criterion value


% Copyrighted code
% (c) Di You, Onur Hamsici and Aleix M Martinez
%
% For additional information contact the authors

K1=exp(-DD/(2*sigma^2));
HH = sum(H);


Q1=sum(sum(A.*K1));

% obtain the mean degree of homoscedasticity between the pairwise subclasses
% in the kernel space
Q2=0;

for i=1:HH-1 
    for j=i+1:HH
        if (sub_label(i) ~= sub_label(j))
            K_i= K1(label==i,label==i);
            K_j= K1(label==j,label==j);
            K_ij=K1(label==i,label==j);
            K_ji=K_ij';
            n1=size(K_i,2);
            n2=size(K_j,2);
            m1 = K_ji - repmat(sum(K_ji,2)./n1,1,n1);
            m2 = K_ij - repmat(sum(K_ij,2)./n2,1,n2);
            t1=sum(sum(m1.*m2'));
            m3 =  K_i - repmat(sum(K_i,2)./n1,1,n1);
            m4 =  K_j - repmat(sum(K_j,2)./n2,1,n2);
            t2=sum(sum(m3.*m3'))+sum(sum(m4.*m4'));
            if t1==0 && t2==0,
                Q2=Q2;
            else
                Q2=Q2+t1/t2;
            end
        end
    end
end

Q2=Q2/(H(1)^2*C*(C-1)/2);  
J=Q1*Q2; 
% minimizing instead of maximizing
J=-J;


