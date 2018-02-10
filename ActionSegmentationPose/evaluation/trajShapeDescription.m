function dst = trajShapeDescription(traj, W)

%%% calculate the shape description of trajectory within the past W frames
n_frames = size(traj, 1);
src = traj(max(n_frames-W, 1):end, :);

%%% use the eigenvectors to describe the shape
C = cov(src);
dst = 2*det(C)/(1e-6+(trace(C))^2);
end

