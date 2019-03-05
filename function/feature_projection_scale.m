function z = feature_projection_scale(x_npca, x_pca, projection_matrix, cos_window)


% 注意这里同时用在了给CNNfeat降维，但是没有改函数名feature_projection_scale


% do the windowing of the output
z = bsxfun(@times, cos_window, projection_matrix * x_pca);
end