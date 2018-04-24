function [xt, xl, xr] = calPatternFromSkeleton(skeleton, type)


if ~strcmp(type, 'jointLocs') && ~strcmp(type, 'relativeAngle') && ~strcmp(type, 'quaternion')

    error('not a valid feature type!');
end



torso_joint = {'ULW', 'OLW','UBW','OBW','UHW','BRK','OHW'};
left_arm_joint = {'SBL','OAL','UAL','HAL','FIL'};
right_arm_joint = {'SBR','OAR','UAR','HAR','FIR'};

%% extract torso pattern xt
xt = []; % location coordinate
rat = []; % relative angle
qut = []; % quaternion



for ii = 1:length(torso_joint)
    idx = find(contains(skeleton.textdata,torso_joint{ii}));
    xt = [xt skeleton.data(:,idx)];     
end

if ~strcmp(type, 'jointLocs')
    
    n_joints = length(torso_joint);
   
    for j = 1:n_joints-2
        p1 = xt(:,3*j-2: 3*j);
        p2 = xt(:,3*(j+1)-2: 3*(j+1));
        p3 = xt(:,3*(j+2)-2: 3*(j+2));
        av = p2-p1;
        bv = p3-p2;
        a = av./repmat(sqrt(sum(av.^2,2))+1e-6,1,3);
        b = bv./repmat(sqrt(sum(bv.^2,2))+1e-6,1,3);
        
        %%% compute relative angles
        c = diag(a*b');
        rat = [rat acos(c)];
        
        %%% compute quaternions
        yaw = acos((a(:,1).*b(:,1)+a(:,2).*b(:,2))./ ( 1e-6+ sqrt(a(:,1).^2+a(:,2).^2).*sqrt(b(:,1).^2+b(:,2).^2) ));
        pitch = acos((a(:,1).*b(:,1)+a(:,3).*b(:,3))./ ( 1e-6+ sqrt(a(:,1).^2+a(:,3).^2).*sqrt(b(:,1).^2+b(:,3).^2) ));
        roll = acos((a(:,2).*b(:,2)+a(:,3).*b(:,3))./ ( 1e-6+ sqrt(a(:,2).^2+a(:,3).^2).*sqrt(b(:,2).^2+b(:,3).^2) ));
        qut = [qut angle2quat(yaw, pitch,roll)];
    end
    
    if strcmp(type,'relativeAngle')
        xt = rat;
    elseif strcmp(type,'quaternion')
        xt = qut;
    end
% else
%     n_joints = size(xt,2)/3;
%     for jj = 1:n_joints
%         xt(:,3*jj-2) = xt(:,3*jj-2)-xt(:,1);
%         xt(:,3*jj-1) = xt(:,3*jj-1)-xt(:,2);
%         xt(:,3*jj-0) = xt(:,3*jj-0)-xt(:,3);
%     end
%     xt(:,1:3) = [];
    
end
        

%% extract left arm pattern xl
xl = []; % location coordinate
ral = []; % relative angle
qul = []; % quaternion

for ii = 1:length(left_arm_joint)
    idx = find(contains(skeleton.textdata,left_arm_joint{ii}));
    xl = [xl skeleton.data(:,idx)];     
end
if ~strcmp(type, 'jointLocs')
    n_joints = length(left_arm_joint);
   
    for j = 1:n_joints-2
        p1 = xl(:,3*j-2: 3*j);
        p2 = xl(:,3*(j+1)-2: 3*(j+1));
        p3 = xl(:,3*(j+2)-2: 3*(j+2));
        av = p2-p1;
        bv = p3-p2;
        a = av./repmat(sqrt(sum(av.^2,2))+1e-6,1,3);
        b = bv./repmat(sqrt(sum(bv.^2,2))+1e-6,1,3);
        
        %%% compute relative angles
        c = diag(a*b');
        ral = [ral acos(c)];
        
        %%% compute quaternions
        yaw = acos((a(:,1).*b(:,1)+a(:,2).*b(:,2))./ ( 1e-6+ sqrt(a(:,1).^2+a(:,2).^2).*sqrt(b(:,1).^2+b(:,2).^2) ));
        pitch = acos((a(:,1).*b(:,1)+a(:,3).*b(:,3))./ ( 1e-6+ sqrt(a(:,1).^2+a(:,3).^2).*sqrt(b(:,1).^2+b(:,3).^2) ));
        roll = acos((a(:,2).*b(:,2)+a(:,3).*b(:,3))./ ( 1e-6+ sqrt(a(:,2).^2+a(:,3).^2).*sqrt(b(:,2).^2+b(:,3).^2) ));
        qul = [qul angle2quat(yaw, pitch,roll)];
    end
    
    if strcmp(type,'relativeAngle')
        xl = ral;
    elseif strcmp(type,'quaternion')
        xl = qul;
    end
% else
%     n_joints = size(xl,2)/3;
%     for jj = 1:n_joints
%         xl(:,3*jj-2) = xl(:,3*jj-2)-xl(:,1);
%         xl(:,3*jj-1) = xl(:,3*jj-1)-xl(:,2);
%         xl(:,3*jj-0) = xl(:,3*jj-0)-xl(:,3);
%     end
%     xl(:,1:3) = [];

end




%% extract right arm pattern xr
xr = []; % location coordinate
rar = []; % relative angle
qur = []; % quaternion

for ii = 1:length(right_arm_joint)
    idx = find(contains(skeleton.textdata,right_arm_joint{ii}));
    xr = [xr skeleton.data(:,idx)];     
end
if ~strcmp(type, 'jointLocs')
    n_joints = length(right_arm_joint);
   
    for j = 1:n_joints-2
        p1 = xr(:,3*j-2: 3*j);
        p2 = xr(:,3*(j+1)-2: 3*(j+1));
        p3 = xr(:,3*(j+2)-2: 3*(j+2));
        av = p2-p1;
        bv = p3-p2;
        a = av./repmat(sqrt(sum(av.^2,2))+1e-6,1,3);
        b = bv./repmat(sqrt(sum(bv.^2,2))+1e-6,1,3);
        
        %%% compute relative angles
        c = diag(a*b');
        rar = [rar acos(c)];
        
        %%% compute quaternions
        yaw = acos((a(:,1).*b(:,1)+a(:,2).*b(:,2))./ ( 1e-6+ sqrt(a(:,1).^2+a(:,2).^2).*sqrt(b(:,1).^2+b(:,2).^2) ));
        pitch = acos((a(:,1).*b(:,1)+a(:,3).*b(:,3))./ ( 1e-6+ sqrt(a(:,1).^2+a(:,3).^2).*sqrt(b(:,1).^2+b(:,3).^2) ));
        roll = acos((a(:,2).*b(:,2)+a(:,3).*b(:,3))./ ( 1e-6+ sqrt(a(:,2).^2+a(:,3).^2).*sqrt(b(:,2).^2+b(:,3).^2) ));
        qur = [qur angle2quat(yaw, pitch,roll)];
    end
    
    if strcmp(type,'relativeAngle')
        xr = rar;
    elseif strcmp(type,'quaternion')
        xr = qur;
    end
% else
%     n_joints = size(xr,2)/3;
%     for jj = 1:n_joints
%         xr(:,3*jj-2) = xr(:,3*jj-2)-xr(:,1);
%         xr(:,3*jj-1) = xr(:,3*jj-1)-xr(:,2);
%         xr(:,3*jj-0) = xr(:,3*jj-0)-xr(:,3);
%     end
%     xr(:,1:3) = [];

end






end
