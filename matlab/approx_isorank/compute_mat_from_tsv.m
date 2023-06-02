function M = compute_mat_from_tsv(file, m, n)
    Ms = tdfread(file);
    cols = fieldnames(Ms);
    p    = cols{1};
    q    = cols{2};
    r    = cols{3};
    M    = sparse(Ms.(p), Ms.(q), Ms.(r), m, n);
end