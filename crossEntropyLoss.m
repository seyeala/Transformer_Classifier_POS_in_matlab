function loss = crossEntropyLoss(y, y_pred)

    y_pred(y_pred == 0) = eps;  
    
    loss = -sum(y .* log(y_pred));
end