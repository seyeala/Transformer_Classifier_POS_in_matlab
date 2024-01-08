function loss = crossentropylossf(y, y_pred)
    % ensure no zero values in y_pred to avoid log(0)
    y_pred(y_pred == 0) = eps;  % eph is MATLAB's smallest positive value
    
    % calculate the cross-entropy loss
    loss = -sum(y .* log(y_pred));
end
