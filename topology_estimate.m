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
    D = zeros(n,k);
    ni = zeros(n,k);

    % Construct neighborhood graph using nearest-neighbour algorithm
    fprintf('Constructing the neighbourhood graph...');
    
    % Compute distances (neighbours)
    sum_Y = sum(Y.^2,2);
    indx=1:n;
    for i=1:n
        DD = bsxfun(@plus,sum_Y',bsxfun(@plus,sum_Y(indx), ...
            -2*(Y(indx,:)*Y')));
        %DD=Y; % maybe try taking just the original distances? but it doesn't work amazingly...

        % sort neighbours
        [DD,ind] = sort(abs(DD),2,'ascend');
        D(indx,:) = sqrt(DD(:,2:k+1));
        ni(indx,:) = ind(:,2:k+1);
    end
    D(D==0) = 1e-9; % ensure no dimension is 0
    DD = sparse(n,n);
    idx  = repmat(1:n,[1 k])';
    DD(sub2ind([n,n],idx,ni(:)))   = D;
    DD(sub2ind([n,n],ni(:),idx))   = D;
    W = DD; % weighted adjacency matrix

    % Select the largest connected component (this is important when the number of effective dimensions > 1)
    blocks = components(W)';
    count = zeros(1,max(blocks));
    for i=1:max(blocks)
        count(i) = length(find(blocks==i));
    end
    [~,block_no] = max(count);
    conn_comp = find(blocks == block_no);
    D = W(conn_comp, conn_comp);
    mapping.D = D;

    % Compute the shortest paths
    disp('Computing shortest paths...');
    D = dijkstra(D, 1:n);
    mapping.DD = D;

    % Reduce dimensionality using eigenvector decomposition
    disp('Constructing low-dimensional embedding...');
    D = D.^2;
    M = -.5 .* (bsxfun(@minus, bsxfun(@minus,D,sum(D,1)'./n), sum(D,1)./n) + sum(D(:))./(n.^2)); % isomap function
    M(isnan(M)) = 0;
    M(isinf(M)) = 0;
    [vec, val] = eig(M);
    if size(vec, 2) < n_dim
        n_dim = size(vec, 2);
        warning(['Target dimensionality reduced to ' num2str(n_dim) '...']);
    end
    % Computing final embedding
    [val, ind] = sort(real(diag(val)), 'descend');
    vec = vec(:,ind(1:n_dim));
    val = val(1:n_dim);
    mappedY = real(bsxfun(@times, vec, sqrt(val)'));

    % Store data
    mapping.conn_comp = conn_comp;
    mapping.k = k;
    mapping.vec = vec;
    mapping.val = val;
    mapping.no_dims = n_dim;
