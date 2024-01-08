function avgLoss = calculateLoss2f(dataset, one_forward, myObjectsMHA, myObjectffnn, classifier)
    totalLoss = 0;
    for i = 1:length(dataset)
        inputVec = dataset{i, 1};
        targetOutput = dataset{i, 2};
        nnOutput = one_forward(inputVec);
        loss = mean(crossEntropyLoss(targetOutput, nnOutput));
        totalLoss = totalLoss + loss;
    end
    avgLoss = totalLoss / length(dataset);
end
