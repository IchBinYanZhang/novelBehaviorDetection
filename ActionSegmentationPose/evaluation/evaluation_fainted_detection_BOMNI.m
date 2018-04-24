function cmat_fainted = evaluation_fainted_detection_BOMNI(gt, prd)


n_frames = length(gt);
%%%% encode gt to state labels %%%%
gt_fainted_ts = find(gt==6);

yt = ones(size(gt_fainted_ts));


%%%% evaluate the detection of abormality%%%%
seg = prd(gt_fainted_ts);
mj_label = mode(seg);




%%%% encode prd to state labels %%%%
ytp = 2*ones(n_frames,1);
changeframe = 1;
j = 1;
used_label = [];
while j <= n_frames
    if j ~= n_frames
        jump = prd(j+1)-prd(j);
        if jump ~= 0  %% a boundary is detected
            seg = prd(changeframe:j);
            mj_label = mode(seg);
            if sum( ismember(used_label, mj_label ))==0 %% new cluster
                ytp(changeframe : j) = 1;
                used_label = [used_label; mj_label];
            end
            changeframe = j;
        end
        
    else
        seg = prd(changeframe:j);
        mj_label = mode(seg);
        if sum( ismember(used_label, mj_label ))==0 %% new cluster
            ytp(changeframe : j) = 1;
            used_label = [used_label; mj_label];
        end
    end
    j = j+1;
end

%%%% calculate precision and recall
ytp_in_fainted_bd = ytp(gt_fainted_ts);
ytp_in_fainted_bd(seg~=mj_label) = 2;




if ytp_in_fainted_bd==yt
    cmat_fainted = length(ytp_in_fainted_bd)*eye(2);
else
    cmat_fainted = confusionmat(yt, ytp_in_fainted_bd);
end
end