function [matching] = find_mapping(R,n1,n2)
% Useing pairwise score R to find the matching

matching =zeros(2,min(n1,n2));

for i=1:min(n1,n2)

    %max_temp = max(max(R));
    %[matching(1,i),matching(2,i)] = find(R==max_temp);
    
    [~,index] = max(R,[],'all','linear');
    matching(2,i) = ceil(index/n2);
    matching(1,i) = index - (matching(2,i)-1)*n2;

    R(matching(1,i),:) = 0;
    R(:,matching(2,i)) = 0;

end
end