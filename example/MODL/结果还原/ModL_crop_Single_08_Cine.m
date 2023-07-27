clear
clc
close all


folders_path = '/media/nas/Seafile_Storage/Raw_data/MICCAIChallenge2023/ChallengeResult/Submission_MoDL/Cine/SingleCoil/Cine/ValidationSet/AccFactor08';  % 指定包含文件夹的父文件夹路径
gt_fpth = '/home/wangcy/Data/MICCAIChallenge2023/GroundTruth/SingleCoil/Cine/ValidationSet/FullSample';
file_names = dir(folders_path);  % 获取父文件夹下的所有文件和文件夹信息

for i = 3:length(file_names)
    folder_name = file_names(i).name;  % 获取文件夹名字
    folder_path = strcat(folders_path, '/',folder_name);  % 构建文件夹的完整路径
    gt_path = strcat(gt_fpth,'/',folder_name);
    
    lax_file_path = strcat(folder_path, '/lax.mat');  % lax.mat文件路径
    sax_file_path = strcat(folder_path, '/sax.mat');  % sax.mat文件路径
    lax_gt_path = strcat(gt_path, '/cine_lax.mat');  % lax.mat文件路径
    sax_gt_path = strcat(gt_path, '/cine_sax.mat');  % sax.mat文件路径
    
%     disp(lax_file_path)
%     disp(sax_gt_path)
    
    % 读取lax.mat文件
%     if exist(lax_file_path, 'file')
%         filetype = 'cine_lax';
%         disp(lax_file_path)
%         load(lax_file_path);
%         load(lax_gt_path);
%         [kx,ky,kz,kt] = size(kspace_single_full);
%         % 处理lax数据
%         %             img4ranking = run4Ranking(data,filetype);
%         if kz<3
%             tmp = data(:,:,:,1:3);            
%         else
%             tmp = data(:,:,round(kz/2)-1:round(kz/2),1:3);        % crop the middle 1/6 of the original image for ranking
%         end
%         reconImg = crop(tmp,[kx,ky,2,3]);
%         [sx,sy,sz,t] = size(reconImg);
%         img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),2,3]));
%         if exist(strcat(folder_path,'/','cine_lax.mat'), 'file')
%             delete(strcat(folder_path,'/','cine_lax.mat'));
%         end
%         save(strcat(folder_path,'/','cine_lax.mat'),'img4ranking');
%         
%     end
%     
    % 读取sax.mat文件
    if exist(sax_file_path, 'file')
        filetype = 'cine_sax';
        load(sax_file_path);
    disp(sax_gt_path)
        load(sax_gt_path);
        [kx,ky,kz,kt] = size(kspace_single_full);
        % 处理sax数据
        %             img4ranking = run4Ranking(sax_data,filetype);
        if kz<3
            tmp = data(:,:,:,1:3);
        else
            tmp = data(:,:,round(kz/2)-1:round(kz/2),1:3);
            % crop the middle 1/6 of the original image for ranking
        end
        reconImg = crop(tmp,[kx,ky,2,3]);
        [sx,sy,sz,t] = size(reconImg);
        img4ranking = single(crop(abs(reconImg),[round(sx/3),round(sy/2),2,3]));
        if exist(strcat(folder_path,'/','cine_sax.mat'), 'file')
            delete(strcat(folder_path,'/','cine_sax.mat'));
        end
        save(strcat(folder_path,'/','cine_sax.mat'),'img4ranking');
        
        clear img4ranking reconImg data
    end
end






