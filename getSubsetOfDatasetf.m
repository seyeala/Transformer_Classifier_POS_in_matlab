function subset = getSubsetOfDatasetf(dataset, numDataPoints)
    %  takes a dataset and a number
    % and returns a subset of the dataset its useful to for quick test of training loops
    if numDataPoints > size(dataset, 1)
        error('numDataPoints exceeds the size of the dataset.');
    end
    subset = dataset(1:numDataPoints, :);
end
