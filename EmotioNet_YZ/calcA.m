function [A,label,sub_label] = calcA(H,nh,l,C)

HH = sum(H);

Q=zeros(l,size(nh,2));
start=0;
for i=1:size(nh,2)
    
    for j=start+1:start+nh(i)
        Q(j,i)=1/nh(i);
    end
    start=sum(nh(1:i));
end

sub_label=zeros(1,size(nh,2));
for i=1: size(nh,2)
    for class = 1:C
        if (i <= sum(H(1:class)))
            sub_label(i) = class;
            break;
        end
    end
    
end
label=zeros(1,l);
for i=1:l
    for j = 1:HH
        if (i <= sum(nh(1:j)))
            label(i) = j;
            break;
        end
    end
end
% obtain the sum of distances between the pairwise subclasses in the kernel
% space
% A= zeros(l,l);
% for i=1:size(nh,2)-1
%     for j=i+1:size(nh,2)
%         if (sub_label(i) ~= sub_label(j))
%           
%             A  = A +  (nh(i)/l)*(nh(j)/l)*(Q(:,i)-Q(:,j))*(Q(:,i)-Q(:,j))';
%         end
%     end
% end
Qt = Q;% * diag(nh)/(l^2);
A = zeros(l,l);
idxTmp = 1: size(Qt,2);
for k1 = 1 : size(Qt,2)
    nhTmp = nh(sub_label~=sub_label(k1) & idxTmp > k1);
    Qt2 = Qt(:,sub_label~=sub_label(k1) & idxTmp > k1);
    Qt2 = Qt2 - repmat(Qt(:,k1),1,size(Qt2,2));
    Qt2 = Qt2*sqrt(diag(nhTmp)*nh(k1)/(l^2));
    A = A + Qt2*Qt2';
end