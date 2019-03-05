function z = feature_projection(x_npca, x_pca, projection_matrix, cos_window)

% get dimensions
[height, width] = size(cos_window);
[num_pca_in, num_pca_out] = size(projection_matrix);


z = bsxfun(@times, cos_window, reshape(x_pca * projection_matrix, [height, width, num_pca_out]));
end                                   % 注意降维完输出的还是原来三维的特征 w * h * dimemsion