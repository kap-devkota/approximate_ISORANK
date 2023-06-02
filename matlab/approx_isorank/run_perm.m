
%---------------------------------
% two graphs for graph matching
%---------------------------------
% generate first graph
n = 2500; % size of the graph

% Erdos Reyni
%prob = 4*log(n)/n; % probabily of edges
%[G1, A1] = Erdos_Reyni_Random_Graph(n, prob);

% B-A model
seed = [0 1 0 0 1;1 0 0 1 0;0 0 0 1 0;0 1 1 0 0;1 0 0 0 0];
A1 = SFNG(n, 5, seed);
G1 = graph(A1,'omitselfloops');

d1 = sum(A1,2);

fprintf("Number of edges = %d\n", nnz(A1)/2);

%figure(1);
%plot(G1);

% permute to get the second grapp
permutation = randperm(n);
G2 = reordernodes(G1, permutation);
A2 = adjacency(G2);
d2 = sum(A2,2);

%figure(2);
%plot(G2);

% generate E (small perturbation of the correct permutation)
E = sparse(1:n,permutation,ones(n,1),n,n);
epsilon = 0.7;
E = full(E) + (-epsilon + (2*epsilon).*rand(n,n));
E = abs(E);
E = E/sum(reshape(E,n^2,1));

% parameters
alpha = 0.6;
maxIt = 100;

% initial guess
R = rand(n*n,1);
R = R/sum(R);
R = reshape(R,n,n);

%----------------------------------
% IsoRank (original version, slow)
%----------------------------------
% R = IsoRank(A1, d1, A2, d2, E, R, alpha, maxIt);

%----------------------------------
% IsoRank (original version but using properties of Kronecker product)
%----------------------------------
tic;
R = IsoRank_kron(A1, d1, A2, d2, E, R, alpha, maxIt);
toc;

%----------------------------------
% Approximate IsoRank 
%----------------------------------
tic;
[R0, SSD] = IsoRank_approx(A1, d1, A2, d2, E, alpha);
toc;
tic;
R1 = IsoRank_kron(A1, d1, A2, d2, E, R0, alpha, 1);
toc;

% measure error
%A = kron(A1,A2);
d = kron(d1,d2);
%D = spdiags(d, 0, n1*n2, n1*n2);
%dtotal = sum(d);
invD = spdiags(1./d, 0, n*n, n*n);
%P = A*invD;
%invDhalf = spdiags(1./sqrt(d), 0, n1*n2,n1*n2);
%tA = invDhalf*A*invDhalf;
%tSSD = invDhalf*reshape(SSD,n1*n2,1);
%tE = invDhalf*reshape(E,n1*n2,1);
%v1 = sqrt(dtotal)*tSSD;
%mu1 = v1'*tE;

err = reshape((R-R0),n*n,1);
fprintf("l2 norm error of R0 = %e\n",norm(err));
fprintf("Weighted norm error of R0 = %e\n",sqrt(err'*invD*err));

err1 = reshape((R-R1),n*n,1);
fprintf("l2 norm error of R1 = %e\n",norm(err1));
fprintf("Weighted norm error of R1 = %e\n",sqrt(err1'*invD*err1));

% accuracy
matching_IsoRank = find_mapping(R,n,n);
[~, order]=sort(matching_IsoRank(1,:));
permutation_IsoRank = matching_IsoRank(2,order);
accuracy_IsoRank = (sum(permutation_IsoRank == permutation))/n;
fprintf("IsoRank accuracy = %f\n",accuracy_IsoRank);

matching_IsoRank_approx = find_mapping(R0,n,n);
[~, order]=sort(matching_IsoRank_approx(1,:));
permutation_IsoRank_approx = matching_IsoRank_approx(2,order);
accuracy_IsoRank_approx = (sum(permutation_IsoRank_approx == permutation))/n;
fprintf("Approximate IsoRank accuracy of R0 = %f\n",accuracy_IsoRank_approx);

matching_IsoRank_approx_1 = find_mapping(R1,n,n);
[~, order]=sort(matching_IsoRank_approx_1(1,:));
permutation_IsoRank_approx_1 = matching_IsoRank_approx_1(2,order);
accuracy_IsoRank_approx_1 = (sum(permutation_IsoRank_approx_1 == permutation))/n;
fprintf("Approximate IsoRank accuracy of R1 = %f\n",accuracy_IsoRank_approx_1);

