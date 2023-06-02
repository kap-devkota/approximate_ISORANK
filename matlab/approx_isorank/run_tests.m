function [T0, T1, Titer, norm0, norm1, accuracy_IsoRank_approx_0, accuracy_IsoRank_approx_1] = run_tests(org1, org2, pairs, alpha, iter, outfolder)

    o1 = split(org1, "/"); o1=o1{end}
    o2 = split(org2, "/"); o2=o2{end}

    tic; 
    R0 = IsoRank_approx(org1, org2, pairs, alpha);
    T0 = toc
    
    tic;
    R1 = IsoRank_kron(org1, org2, pairs, alpha, 1);
    T1 = toc
    
    tic;
    R = IsoRank_kron(org1, org2, pairs, alpha, iter);
    Titer = toc
    
    m = read_json_dim(org1 + ".json");
    n = read_json_dim(org2 + ".json");
    

    norm0 = norm(reshape(R0 - R, m * n, 1))
    norm1 = norm(reshape(R1 - R, m * n, 1))
    
    %matching_IsoRank_approx_0 = find_mapping(R0,m,n);
    %matching_IsoRank_approx_1 = find_mapping(R1,m,n);
    %matching_IsoRank          = find_mapping(R,m,n);
    
    % Saving matrices
    fprintf("Saving\n")
    saveiterloc = sprintf("%s/%s_%s_%f_%d.mat", outfolder, o1, o2, alpha, 0);
    save(saveiterloc, "R0");
    saveiterloc = sprintf("%s/%s_%s_%f_%d.mat", outfolder, o1, o2, alpha, 1);
    save(saveiterloc, "R1");
    saveiterloc = sprintf("%s/%s_%s_%f_%d.mat", outfolder, o1, o2, alpha, iter);
    save(saveiterloc, "R");
    
    %[~, order]=sort(matching_IsoRank(1,:));
    %perm_Isorank = matching_IsoRank(2, order); 
    
    %[~, order] = sort(matching_Isorank_approx_0(1, :));
    %perm_Approx0 = sort(matching_Isorank_approx_0(2, order));
    
    
    %[~, order] = sort(matching_Isorank_approx_1(1, :));
    %perm_Approx1 = sort(matching_Isorank_approx_1(2, order));

    %accuracy_IsoRank_approx_0 = (sum(perm_Approx0 == perm_Isorank))/min([m, n]);
    %accuracy_IsoRank_approx_1 = (sum(perm_Approx1 == perm_Isorank))/min([m, n]);
    
    f = fopen(outfolder + "/output_norms.txt", "a+");
    fprintf(f, "%f %f %f %f %f %s %s %f %d\n", T0, T1, Titer, norm0, norm1, org1, org2, alpha, iter);
    fclose(f);
end
