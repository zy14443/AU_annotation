function J = MaxhomoCrazy(H,label,sub_label, C, A, sigma,DD)

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

K1 = exp(-DD/(2*sigma^2));
HH = sum(H);


Q1 = sum(sum(A.*K1));

% obtain the mean degree of homoscedasticity between the pairwise subclasses
% in the kernel space
Q2 = 0;
b = length(unique(label));
Knew = K1;
% for i=1:HH-1 
minD = inf;
a = 0;
idx = 0;
for k1 = 1 : HH - 1,
    for k2 = k1 + 1 : HH,
        K_j = Knew(label==k2,label==k2);
        K_i = Knew(label==k1,label==k1);
        K_ij = Knew(label==k1,label==k2);        
        tmp = mean(K_j(:)) + mean(K_i(:)) - 2 * mean(K_ij(:));
%         tmp = tmp/size(K_i,2);
        a = a + tmp;
        idx = idx + 1;
        if tmp < minD,
            minD = tmp;
        end
    end
end

% a = sqrt(minD)/b;
% a = 1e-12;%sqrt(a/idx)/b;
a = a/idx/b;
% a = 1./a;

Kcell = cell(1,HH);
traceVec = zeros(1,HH);
for k1 = 1 : HH,
    Ktmp = K1(label==k1,:)';
    n1=size(Ktmp,2);
    Kcell{k1} = Ktmp*(eye(size(Ktmp,2))-ones(size(Ktmp,2))/n1)*Ktmp'/n1;
    traceVec(k1) = sum(Kcell{k1}(:).^2);
end

for i=1:HH
   
        
    t1 = trace(Kcell{i}*K1*a);
    t2 = a^2*sum(K1(:).^2) + traceVec(i);
%         m3 =  K_i - repmat(sum(K_i,2)./n1,1,n1);
   
%             l = size(m4,1);
        %         t2=sum(sum(m3.*m3'))+sum(sum(m4.*m4'));
        %     t2 = l*a^2 + trace(m4^2);
    if t1==0 && t2==0,
        Q2 = Q2;
    else
        Q2 = Q2 + t1/t2;
    end

end

Q2 = Q2 / (H(1)^2 * C * (C - 1) / 2);  
J = Q1 * Q2*1e20 ; 
% J = Q2*1e20;
% minimizing instead of maximizing
J = -J;


% for i=1:HH-1 
%     for j=i+1:HH
%         if (sub_label(i) ~= sub_label(j))
% %     if (sub_label(i) ~= sub_label(j))
% %         K_i= K1(label==i,label==i);
% %     K_j= K1(label==j,label==j);
% %             K_i= K1(label==i,:)';
% %             K_j= K1(label==j,:)';
% % %         K_ij=K1(label==i,label==j);
% % %         K_ji=K_ij';
% % %         n1=size(K_i,2);
% %             n2=size(K_j,2);
% %             n1=size(K_i,2);
% % %         m1 = K_ji - repmat(sum(K_ji,2)./n1,1,n1);
% % %         m2 = K_ij - repmat(sum(K_ij,2)./n2,1,n2);
% % %     m4 =  K_j - repmat(sum(K_j,2)./n2,1,n2);
% %             m3 = K_i*(eye(size(K_i,2))-ones(size(K_i,2))/n1)*K_i'/n1;
% %             m4 = K_j*(eye(size(K_j,2))-ones(size(K_j,2))/n2)*K_j'/n2;
% %     t1 = trace(m4)*a;%sum(sum(m4))*a;
%             t1 = sum(sum(Kcell{i}.*Kcell{j}'));
%             t2 = traceVec(i) + traceVec(j);
% %         m3 =  K_i - repmat(sum(K_i,2)./n1,1,n1);
%    
% %             l = size(m4,1);
%         %         t2=sum(sum(m3.*m3'))+sum(sum(m4.*m4'));
%         %     t2 = l*a^2 + trace(m4^2);
%             if t1==0 && t2==0,
%                 Q2 = Q2;
%             else
%                 Q2 = Q2 + t1/t2;
%             end
%         end
%     end
% end