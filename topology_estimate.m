function [mappedY, mapping] = topology_estimate(Y,n_dim,k)
% function [mappedY, mapping] = topology_estimate(D,n_dim,k,varargin);
% runs the isomap algorithm to estimate the topology on the given data
%
% INPUT: 
%           - D: data (size n x p; if data raw: n - number of regions, p - number of
%                dimensions, e.g. voxel activation; if data distance - p=n)
%           - n_dim: number of dimensions to reduce the data to
%           - k: number of neihbours to consider in the nearest-neighbour
%               graph computation
%
% OUTPUT:
%           - mappedY: datapoints mapped onto the lower-dimensional
%               topology
%           - mapping: structure providing additional info on the mapping
%
% example usage: [X,mp] = topology_estimate(data,3,10)
%
%--------------------------------------------------------------------------
    n = size(Y,1);
    ni = zeros(n,k);

    %% 1) Construct neighborhood graph using nearest-neighbour algorithm
    fprintf('Constructing the neighbourhood graph...');
    
    % Compute distances (neighbours)
    sum_Y = sum(Y.^2,2);
    indx=1:n;
    N = bsxfun(@plus,sum_Y',bsxfun(@plus,sum_Y(indx), ...
        -2*(Y(indx,:)*Y')));
    %DD=Y; % maybe try taking just the original distances? but it doesn't work amazingly...
    
    % sort neighbours
    [NN,ind]    = sort(abs(N),2,'ascend');
    A(indx,:)   = sqrt(NN(:,2:k+1));    % consider only k neighbours
    ni(indx,:)  = ind(:,2:k+1);
    A(A==0)     = 1e-9;                 % ensure no dimension is 0
    W           = sparse(n,n);          % preallocate the adjacency matrix W
    idx         = repmat(1:n,[1 k])';
    W(sub2ind([n,n],idx,ni(:)))   = A;  % weighted adjacency matrix
    W(sub2ind([n,n],ni(:),idx))   = A;

    % Select the largest connected component (this is important when the number of effective dimensions > 1)
    blocks = components(W)';
    count = zeros(1,max(blocks));
    for i=1:max(blocks)
        count(i) = length(find(blocks==i));
    end
    [~,block_no]    = max(count);
    conn_comp       = find(blocks == block_no);
    D               = W(conn_comp, conn_comp);
    mapping.D       = D;

    %% 2) Compute the shortest paths
    disp('Computing shortest paths...');
    P = dijkstra(D, 1:n);
    mapping.DD = P; % geodesic distances between elements (dissimilarity-like measure)

    % Reduce dimensionality using eigenvector decomposition
    disp('Constructing low-dimensional embedding...');
    E = P.^2;
    M = -.5 .* (bsxfun(@minus, bsxfun(@minus,E,sum(E,1)'./n), sum(E,1)./n) + sum(E(:))./(n.^2)); % isomap function - back to similarity
    M(isnan(M)) = 0;
    M(isinf(M)) = 0;
    [vec, val]  = eig(M);
    if size(vec, 2) < n_dim
        n_dim = size(vec, 2);
        warning(['Target dimensionality reduced to ' num2str(n_dim) '...']);
    end
    %% 3) Construct low-dimensional embedding
    [val, ind]  = sort(real(diag(val)), 'descend');
    vec         = vec(:,ind(1:n_dim));
    val         = val(1:n_dim);
    mappedY     = real(bsxfun(@times, vec, sqrt(val)'));

    % Store data
    mapping.conn_comp   = conn_comp;
    mapping.eigVec      = vec;
    mapping.eigVal      = val;
    mapping.n_dim       = n_dim;
    mapping.k           = k;
