function [R] = IsoRank_kron(file1, file2, pairs, alpha, maxIt)
%function [R] = IsoRank_kron(A1, d1, A2, d2, E, R, alpha, maxIt)
% IsoRank using properties of Kronecker product and do not form P
% explicitly


m = read_json_dim(file1 + ".json");
n = read_json_dim(file2 + ".json");
[A1, d1] = compute_AD(file1 + ".a.tsv", m);
[A2, d2] = compute_AD(file2 + ".a.tsv", n);
E = compute_mat_from_tsv(pairs, m, n);
E = E/sum(reshape(E,n*m,1));

[R, SSD] = IsoRank_approx(file1, file2, pairs, alpha);

% get size
n1 = size(A1,1);
n2 = size(A2,1);

% transition matrices
invD1 = spdiags(1./d1, 0, n1, n1);
P1 = A1*invD1;

invD2 = spdiags(1./d2, 0, n2, n2);
P2 = A2*invD2;

% reshape E (n1xn2 -> n2xn1)
E = reshape(reshape(E,n1*n2,1), n2,n1);
normE = norm(E,'fro');

% reshape R (n1xn2 -> n2xn1)
R = reshape(reshape(R,n1*n2,1), n2,n1);

% main loop (using power method)
for k = 1:maxIt
    R_old = R;
    R = (1-alpha)*(P2*R*P1') + alpha*E;
    %sum(R)
    if (norm(R_old-R,'fro')/normE)<1e-12
        fprintf("IsoRank Power iteration stops at the %d-th step\n", k);
        break;
    end
end

% reshape R (n2xn1 -> n1*n2)
R = reshape(reshape(R, n1*n2, 1),n1,n2);

end