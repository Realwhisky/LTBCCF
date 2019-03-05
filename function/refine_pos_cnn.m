
function [pos, max_response]=refine_pos_cnn(imCNN, pos, indLayers, nweights, CNNpadwindow_sz, cos_windowCNNpad, CNNf_num, CNNfenmu_den,...
          lambda, yCNN, CNNpadcos_sz, projection_matrixnn, featureRatio, currentScaleFactor, interpolate_response, maxresponseMIX)    
            
    
                feat = extractFeature(imCNN, pos, CNNpadwindow_sz,cos_windowCNNpad, indLayers);
    
                conv_1=(feat{1});       % 原本是single类型的conv_1=single(feat{1});现在保持原数！
                conv_2=(feat{2});
                conv_3=(feat{3});
                temp_CNN1 = (conv_1(:,:,1:512));
                temp_CNN2 = (conv_2(:,:,1:512));
                temp_CNN3 = (conv_3(:,:,1:256));
                temp_CNN1feat= reshape(temp_CNN1, [size(temp_CNN1, 1)*size(temp_CNN1, 2), size(temp_CNN1, 3)]);  % 重塑成2维
                temp_CNN2feat= reshape(temp_CNN2, [size(temp_CNN2, 1)*size(temp_CNN2, 2), size(temp_CNN2, 3)]);
                temp_CNN3feat= reshape(temp_CNN3, [size(temp_CNN3, 1)*size(temp_CNN3, 2), size(temp_CNN3, 3)]);        
                temp_CNN1feat_proj=feature_projection( [] , temp_CNN1feat, projection_matrixnn{1}, cos_windowCNNpad);% CNN特征降维
                temp_CNN2feat_proj=feature_projection( [] , temp_CNN2feat, projection_matrixnn{2}, cos_windowCNNpad);
                temp_CNN3feat_proj=feature_projection( [] , temp_CNN3feat, projection_matrixnn{3}, cos_windowCNNpad);
                temp_CNN1feat_projf = fft2(temp_CNN1feat_proj);         % 降维特征转换到傅里叶域
                temp_CNN2feat_projf = fft2(temp_CNN2feat_proj);
                temp_CNN3feat_projf = fft2(temp_CNN3feat_proj);
    
                tempCNNfeat=cell(3,1);                               
                tempCNNfeat{1}=temp_CNN1feat_projf;         % 投影特征放到cell里，方便计算1:3
                tempCNNfeat{2}=temp_CNN2feat_projf;
                tempCNNfeat{3}=temp_CNN3feat_projf;
    
                nweights  = reshape(nweights,1,1,[]);                        % nweight调整成三层的形式，方便做bsxfun@times
                responseCNN=cell(1,3);                                       % 存储每一层输出层响应
                response_layers=zeros([CNNpadcos_sz, length(indLayers)]);   % 定义存放3个输出层响应的变量
                temp_CNNresponse=cell(1,3);
        
                for ii=1:length(indLayers)
                    responseCNN{ii} = sum(CNNf_num{ii} .* tempCNNfeat{ii}, 3) ./ (CNNfenmu_den{ii} + lambda);
                    temp_CNNresponse{ii} =real(ifft2(responseCNN{ii}));            
                    response_layers(:,:,ii) = temp_CNNresponse{ii}/max(temp_CNNresponse{ii}(:));
                    response_layers(:,:,ii) = responseCNN{ii};
                end   
    % 这里归一化一下响应值responseCNNmix！！！（就是找响应的最大的那个，除以它，最大的值就是1）；    
                responseCNNmix = sum(bsxfun(@times, response_layers, nweights), 3);
                responsefCNNmix_aver =responseCNNmix/1.75;                     % (1/1.75)消除三个层叠加的1+0.5+0.25权重
                
            if interpolate_response > 0
                if interpolate_response == 2
                   interp_szCNN = size(yCNN) * featureRatio * currentScaleFactor;
                end
                   interp_szCNN = size(yCNN) * featureRatio;
                   responsefCNN = resizeDFT2(responsefCNNmix_aver, interp_szCNN); % 有条件地利用CNN特征
            end
            
                   responseCNN = ifft2(responsefCNN, 'symmetric');       % 归一化CNN特征                 
                   
                   
                   
                   [row, col] = find(responseCNN == max(responseCNN(:)), 1);
                   disp_row = mod(row - 1 + floor((interp_szCNN(1)-1)/2), interp_szCNN(1)) - floor((interp_szCNN(1)-1)/2);
                   disp_col = mod(col - 1 + floor((interp_szCNN(2)-1)/2), interp_szCNN(2)) - floor((interp_szCNN(2)-1)/2);
                   
               if  interpolate_response == 1
                   translation_vec = round([disp_row, disp_col] * currentScaleFactor);
               end
                   
                   max_response = max(responseCNN(:));                      % 这里是求得的最大响应；
%                    if max_response<1.1*maxresponseMIX,
%                       pos = pos + 0;
%                    else
                   pos = pos + translation_vec;                             % 检测器重新确定的位置
%                    end
                   
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
              
