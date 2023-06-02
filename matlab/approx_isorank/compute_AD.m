function [A, d] = compute_AD(file, dim)
    Astr = tdfread(file);
    cols = fieldnames(Astr);
    p    = cols{1};
    q    = cols{2};
    p_   = Astr.(p);
    q_   = Astr.(q);
    A    = sparse(cat(1, p_, q_), cat(1, q_, p_), 1, dim, dim);
    d    = sum(A, 2);
end