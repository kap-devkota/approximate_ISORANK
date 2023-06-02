function [R, SSD] = IsoRank_approx(file1, file2, pairs, alpha)
% function [R, SSD] = IsoRank_approx(A1, d1, A2, d2, E, alpha)
% Approximate IsoRank
m = read_json_dim(file1 + ".json");
n = read_json_dim(file2 + ".json");
[A1, d1] = compute_AD(file1 + ".a.tsv", m);
[A2, d2] = compute_AD(file2 + ".a.tsv", n);
E = compute_mat_from_tsv(pairs, m, n);
E = E/sum(reshape(E,n*m,1));
% get size
n1 = size(A1,1);
n2 = size(A2,1);

% get d of the tensor graph
%d = kron(d1,d2);
d = kron(d2,d1);

% get steady-state distribution of the tensor graph
SSD = d/sum(d);
SSD = reshape(SSD, n1, n2);

% compute the approximation
R = (1-alpha)*SSD + alpha*E;

end