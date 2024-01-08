

function vectorsWithPosition = positionalEncodingf(vectors, d_model)
    [maxTokens, d_model] = size(vectors);
    PE = zeros(maxTokens, d_model);

    for pos = 1:maxTokens
        for i = 0:(d_model/2 - 1)
            PE(pos, 2*i+1) = sin(pos / (10000^(2*i/d_model)));
            PE(pos, 2*i+2) = cos(pos / (10000^(2*i/d_model)));
        end
    end
    vectorsWithPosition = vectors + PE;
end
