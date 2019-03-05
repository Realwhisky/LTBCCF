function [ window_sz, app_sz ] = search_window( target_sz, im_sz, config)

% ********* ����ʵ������������Ҳ����padding����С�����ã�����target_sz�Ĵ�С�� *********

% ���ڸ߶Ƚϴ�Ķ��󣬽�������������Ϊ1.4���߶ȡ�
       
    if (target_sz(1)/target_sz(2)<=2.4 && target_sz(1)/target_sz(2) >2)...
            || (target_sz(1)/target_sz(2)<=3.1 &&target_sz(1)/target_sz(2)>2.8)
        window_sz = floor(target_sz.*[1.4,1+config.padding ]);        
        % aaa=target_sz(1)/target_sz(2);   1.2*aaa
        
% target_sz = [ground_truth(1,4), ground_truth(1,3)]��config.padding=1.8��ǰ��Ԥ��ģ�
% Ŀ���/��>2�Ļ���������window_sz=����ȡ������*1.4����*��1+1.8����(������һ�㣬�����һ��)

% ���ڸ߶ȺͿ�ȶ��Ƚϴ�����壬��������ռ����ͼ���10%��ֻ����2���ĸ߶ȺͿ��
    elseif (target_sz(1)/target_sz(2)>2.4 && target_sz(1)/target_sz(2)<=2.8)...
            || target_sz(1)/target_sz(2)>3.1
        window_sz = floor(target_sz.*[1.4,2+config.padding ]); 
        
    elseif min(target_sz)>80 && prod(target_sz)/prod(im_sz(1:2))>0.1
        window_sz=floor(target_sz*2);
    % ���� ����Сtarget_sz/im_szҲ����ͼ���С�ı���������10%��������Сtarget_sz����߸ߴ���80��
    % ���Ǿ�ȡ2���ĸߺͿ�ȵ���������
    % ���򣬾����Լ����õ�padding��������

    else        
        window_sz = floor(target_sz * (1 + config.padding));
    end

    app_sz=target_sz+2*config.features.cell_size;   % �߶ȴ����ִ�С���䣬������cell_size��Ϊ�˼���HOG����ʱҪȥ��
                                                    % ����ϵ���������һ������һ����cell��������ǰ�������õ���
    
    % config.feature.cell_size ΪHOG���������С����run_tracker������Ϊ4,�Ҷ�Ϊ1
    % ��������app_size=target_sz + 8 ���� target_sz + 2

end



