function dataset = process_text_data2f(csv_file, maxTokens, wordVectorMap, d_model)
    % read the CSV
    data = readtable(csv_file, 'ReadVariableNames', true);

    % calculate the  average vector for unknown words
    wordVectorsCellArray = values(wordVectorMap);
    avgVector = mean(cell2mat(wordVectorsCellArray'), 1);

    % define the mapping from POS tags to categories
    noun_tags = [21, 24, 22, 23, 25, 28, 29];
    verb_tags = [37, 38, 39, 40, 41, 42];
    adj_adv_tags = [16, 17, 18, 30, 31, 32];

    % initialize dataset
    dataset = cell(size(data, 1), 2);

    for i = 1:size(data, 1)
        % split tokens by space
        tokens = strsplit(data.tokens{i});
        % initialize vectors array for this row
        vectors = zeros(maxTokens, size(wordVectorMap('start'), 2));
        tokenCount = min(maxTokens, length(tokens));

        for j = 1:tokenCount
            token = lower(regexprep(tokens{j}, '[^a-zA-Z0-9_-]', ''));
            if isKey(wordVectorMap, token)
                vectors(j, :) = wordVectorMap(token);
            else
                vectors(j, :) = avgVector;
            end
        end

        pos_tags = eval(data.pos_tags{i});
        pos_tags = pos_tags(1:tokenCount); % truncate pos_tags  match maxTokens

        % map tags to one-hot encodin g
        categories = zeros(tokenCount, 4); %  initialize categories
        for k = 1:tokenCount
            if ismember(pos_tags(k), noun_tags)
                categories(k, 1) = 1;
            elseif ismember(pos_tags(k), verb_tags)
                categories(k, 2) = 1;
            elseif ismember(pos_tags(k), adj_adv_tags)
                categories(k, 3) = 1;
            else
                categories(k, 4) = 1; % others c ategories
            end
        end

        % repeat if they are shorter than maxTokens
        if tokenCount < maxTokens
            repeatTimes = ceil(maxTokens / tokenCount);
            vectors = repmat(vectors, repeatTimes, 1);
            vectors = vectors(1:maxTokens, :);
            categories = repmat(categories, repeatTimes, 1);
            categories = categories(1:maxTokens, :); %length is maxTokens
        end

        % add positional e ncoding
        vectorsWithPosition = positionalEncodingf(vectors, d_model);
        % store  dataset
        dataset{i, 1} = vectorsWithPosition;
        dataset{i, 2} = categories;
    end
end
