function [angles, quaternions] = calRotationFromPosition(skeleton)

triple_connect = [13 1 17; 1 2 3; 2 3 4; 3 5 6; 5 6 7; 6 7 8; 3 9 10;
    9 10 11; 10 11 12; 1 13 14; 13 14 15; 14 15 16; 1 17 18; 17 18 19; 18 19 20];

n_samples = size(triple_connect,1);
angles = zeros(n_samples,1);
quaternions = zeros(n_samples,4);

for ii = 1:n_samples
    ptset = triple_connect(ii,:);
    p1 = skeleton(ptset(1),:);
    p2 = skeleton(ptset(2),:);
    p3 = skeleton(ptset(3),:);
    av = p2-p1;
    bv = p3-p2;
    a = av/(norm(av)+1e-6);
    b = bv/(norm(bv)+1e-6);
    
    %%% cal relative angle
    v = cross(a,b);
    c = a*b';
    s = norm(v);
    angles(ii) = acos(c);
    
    %%% cal yaw,pitch,roll
    yaw = acos((a(1)*b(1)+a(2)*b(2))/ ( 1e-6+ norm([a(1) a(2)])*norm([b(1) b(2)]) ));
    pitch = acos((a(1)*b(1)+a(3)*b(3))/ ( 1e-6+norm([a(1) a(3)])*norm([b(1) b(3)]) ));
    roll = acos((a(2)*b(2)+a(3)*b(3))/ ( 1e-6+norm([a(2) a(3)])*norm([b(2) b(3)]) ));
        
    quaternions(ii,:) = angle2quat(yaw, pitch,roll);
end

