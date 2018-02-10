function label = spectralClustering(varargin)


if nargin==3
    X = varargin{1};
    sigma = varargin{2};
    KK = varargin{3};


    %%% create fully connected graph and perform Kmeans clustering
    %%% depending on the selected K

    dists = pdist(X);
    Z = squareform(dists);
    edge_weights = exp(-Z/sigma) - eye(size(Z));
    node_weights = diag(sum(edge_weights,1))+1e-6;
    % L = node_weights - edge_weights;
    L = eye(size(node_weights,1))-inv(sqrt(node_weights))*edge_weights*inv(sqrt(node_weights));
    [V, spectral] = eig(L);
    spectral = diag(spectral);
    [~,lambda_order] = sort(spectral); % sort eigenvalues in ascending order
    V = V(:,lambda_order);% sort the corresponding eigenvectors
    % 
    % [~, iii] = sort(spectral, 'descend');
    VV = V(:,1:KK);
    % Y = rowNormalisationL2(VV);
    node_weights_vec = sqrt(diag(node_weights));
    Y = VV./ repmat(node_weights_vec, 1, KK);

    YY = rowNormalisationL2(Y);
    YY = real(YY); %%% due to numerical error, we could get X+0.0000i
    if KK ~= 1
        [label] = kmeans(YY, KK);
    else
        label = ones(size(X,1),1);

    end
    
elseif nargin==5
    X = varargin{1};
    sigma = varargin{2};
    KK = varargin{3};
    TT = varargin{4};
    wt = varargin{5};
    %%% create fully connected graph and perform Kmeans clustering
    %%% depending on the selected K
    

    dists = pdist(X);
    Z = squareform(dists);
    TT = TT+1e-6;
    T = wt*(TT'+TT - diag(diag(TT'+TT)))/2;
    edge_weights = exp(-Z/sigma) .* T;
%     edge_weights = T+100;
    node_weights = diag(sum(edge_weights,1));
    % L = node_weights - edge_weights;
    L = eye(size(node_weights,1))-sqrt(inv(node_weights))*edge_weights*sqrt(inv(node_weights));
    [V, spectral] = eig(L);
    % spectral = diag(spectral);
    % 
    % [~, iii] = sort(spectral, 'descend');
    VV = V(:,1:KK);
    % Y = rowNormalisationL2(VV);
    node_weights_vec = sqrt(diag(node_weights));
    Y = VV./ repmat(node_weights_vec, 1, KK);
    YY = rowNormalisationL2(Y);

    if KK ~= 1
        [label] = kmeans(YY, KK);
    else
        label = ones(size(X,1),1);

    end
    
end   
    
end






function Y = rowNormalisationL2(X)
XX = X.^2;
D = sqrt(sum(XX,2)) + 1e-6;
Y = X./repmat(D,1,size(X,2));
end
