function [labels,cut_locs] = calKernelizedTemporalSegment(X, T, sigma_s, sigma_t, cut_method)
%%% Input: X - the input features, arranged by samples X dimensions
%%%        T - the size of timewindow
%%%        sigma_s - the spatio radius of kernel
%%%        sigma_t - the temporal radius of kernel, does not function, since
%%%        the temporal kernel is innerproduct of the local tanglent plane
%%%        cut_method - either 'sequential' or 'batch'
%%% this script implement KTC-S methed introduced in Dian Gong, et al. 2012

local_set = 5; %%% when compute local tagent similarity, use this local set.
[n_samples, n_dims] = size(X);
labels = zeros(n_samples,1);
cut_locs = [];

% if mod(T,2)==0 % time window T should be odd
%     T = T+1;
% end


if strcmp(cut_method, 'sequential')

    %%% main loop
    p = 1;
    ll = 0;
    while(p < n_samples)
        up = min(p+T-1, n_samples);
        Xt = X(p:up, :);
    %     disp('-- compute kernel');
        Kst = calSpatialTemporalKernelMatrix(Xt, sigma_s, sigma_t, local_set);
    %     Kst = real(Kst);
        %%% optimize the energy function via searching
    %     disp('-- optimize..');
        cut = 2;
        loss_max = -10^10;
        e0 = zeros(T,1);
        up2 = min(T, size(Kst,2));
        for ii = 2:up2-1
            k11 = Kst(1:ii, 1:ii);
            k22 = Kst(ii+1:up2, ii+1:up2);
            k12 = Kst(ii+1:up2, 1:ii);
            loss = sum(k11(:))/prod(size(k11)) + sum(k22(:))/prod(size(k22)) ...
                - 2*sum(k12(:))/prod(size(k12));
            if loss > loss_max
                cut = ii;
                loss_max = loss;
            end
        end
        cut = p+cut;
        cut_locs = [cut_locs cut];
        %%% assign labels depending on the cut
        ll = ll+1;
        labels(p:cut) = ll;
        ll = ll+1;
        labels(cut+1:up) = ll;

        p = cut+1;
    %     fprintf('-- cut = %d\n', cut);
        if up==n_samples
            if cut_locs(end)>=n_samples
                cut_locs(end) = [];
            end
            break;
        end
    end
    labels(labels==0) = ll;

elseif strcmp(cut_method, 'batch')
    n_frames = size(X,1);
    n_clusters = round(0.02*n_frames); %; %  36 CMUMAD; round(0.02*n_frames) % TUMKitchen & BOMNI
    Kst = calSpatialTemporalKernelMatrix(X, sigma_s, sigma_t, local_set);
    [oscclusters,~,~] = ncutW(Kst,n_clusters);
    labels = denseSeg(oscclusters, 1);
    cut_locs = find( diff(labels) ~=0 );
    cut_locs = cut_locs';
end
   

end


function Kst = calSpatialTemporalKernelMatrix(X, sigma_s, sigma_t, local_set)


%%% spatial kernel matrix;
Ks = exp(-sigma_s * squareform(pdist(X)));


%%% temporal kernel matrix;
V = calLocalTangentNormal(X, local_set);
Kt= (V*V');

%%% combine them
Kst = Ks.*Kt;
% Kst = Ks;
end


function V = calLocalTangentNormal(X, local_set)

n_samples = size(X,1);
V = [];

%%% mirror the boundary
bd = (local_set-1)/2;
XX = zeros(size(X,1)+2*bd, size(X,2));
XX(bd+1 : bd+n_samples,:) = X;
for ii = 1:bd
    XX(ii,:) = X(1,:);
    XX(end+1-ii,:) = X(end,:);
end

%%% calculate tangent
for ii = 1:n_samples
%     [comp] = pca_1comp( (XX(ii : ii+2*bd, :))');
%     V = [V; comp(:,1)'];
    comp = mean(XX(ii : ii+2*bd, :),1); %% the mean value
    V = [V; comp];
end

V = V./repmat(sqrt(sum(V.*V,2)),1,size(V,2)); % l2-normalize of each row

end
        
