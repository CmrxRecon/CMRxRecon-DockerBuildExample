clear
clc
close all


folders_path = '/media/nas/Seafile_Storage/Raw_data/MICCAIChallenge2023/ChallengeResult/Submission_MoDL/Mapping/SingleCoil/Mapping/ValidationSet/AccFactor10';  % 指定包含文件夹的父文件夹路径
gt_fpth = '/home/wangcy/Data/MICCAIChallenge2023/GroundTruth/SingleCoil/Mapping/ValidationSet/FullSample';
file_names = dir(folders_path);  % 获取父文件夹下的所有文件和文件夹信息

for i = 3:length(file_names)
    folder_name = file_names(i).name;  % 获取文件夹名字
    folder_path = strcat(folders_path, '/',folder_name);  % 构建文件夹的完整路径
    gt_path = strcat(gt_fpth,'/',folder_name);
    
    lax_file_path = strcat(folder_path, '/T1map.mat');  % lax.mat文件路径
    sax_file_path = strcat(folder_path, '/T2map.mat');  % sax.mat文件路径
    lax_gt_path = strcat(gt_path, '/T1map.mat');  % lax.mat文件路径
    sax_gt_path = strcat(gt_path, '/T2map.mat');  % sax.mat文件路径
    
    disp(lax_file_path)
    disp(sax_gt_path)
    
    % 读取lax.mat文件
    if exist(lax_file_path, 'file')
        filetype = 'T1map';
        load(lax_file_path);
        load(lax_gt_path);
        [kx,ky,kz,kt] = size(kspace_single_full);
        % 处理lax数据
        %             img4ranking = run4Ranking(data,filetype);
        if kz<3
            tmp = data;            
        else
            tmp = data(:,:,round(kz/2)-1:round(kz/2),:);        % crop the middle 1/6 of the original image for ranking
        end
        reconImg = crop(tmp,[kx,ky,2,size(data,4)]);
        [sx,sy,sz,t] = size(reconImg);
        if exist(strcat(folder_path,'/','T1map.mat'), 'file')
             delete(strcat(folder_path,'/','T1map.mat'));
         end
        img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),2,t]));
        save(strcat(folder_path,'/','T1map.mat'),'img4ranking');
        
    end
    
%     % 读取sax.mat文件
%     if exist(sax_file_path, 'file')
%         filetype = 'T2map';
%         load(sax_file_path);
%         load(sax_gt_path);
%         [kx,ky,kz,kt] = size(kspace_single_full);
%         % 处理sax数据
%         %             img4ranking = run4Ranking(sax_data,filetype);
%         if kz<3
%             tmp = data;
%         else
%             tmp = data(:,:,round(kz/2)-1:round(kz/2),:);
%             % crop the middle 1/6 of the original image for ranking
%         end
%         reconImg = crop(tmp,[kx,ky,2,3]);
%         [sx,sy,sz,t] = size(reconImg);
%         img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),2,t]));
%         save(strcat(folder_path,'/','T2map.mat'),'img4ranking');
%         
%         clear img4ranking reconImg data
%     end
end






