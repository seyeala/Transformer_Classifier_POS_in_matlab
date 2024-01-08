function class_metrics = calculatemetricsf(dataset, oneforward, myobjectsmha, myobjectsffnn, classifier)
    num_classes = length(myobjectsffnn); 
    class_true_positives = zeros(1, num_classes);
    class_false_positives = zeros(1, num_classes);
    class_false_negatives = zeros(1, num_classes);

    for i = 1:length(dataset)
        testinputvec = dataset{i, 1};
        testtargetoutput = dataset{i, 2};
        testnnoutput = oneforward(testinputvec);
        [~, predictedlabels] = max(testnnoutput, [], 2);
        [~, truelabels] = max(testtargetoutput, [], 2);

        for c = 1:num_classes
            true_positive = sum(predictedlabels == c & truelabels == c);
            false_positive = sum(predictedlabels == c & truelabels ~= c);
            false_negative = sum(predictedlabels ~= c & truelabels == c);

            class_true_positives(c) = class_true_positives(c) + true_positive;
            class_false_positives(c) = class_false_positives(c) + false_positive;
            class_false_negatives(c) = class_false_negatives(c) + false_negative;
        end
    end

    class_accuracies = class_true_positives ./ (class_true_positives + class_false_positives);
    class_precisions = class_true_positives ./ (class_true_positives + class_false_positives);
    class_recalls = class_true_positives ./ (class_true_positives + class_false_negatives);

    class_precisions(isnan(class_precisions)) = 0;
    class_recalls(isnan(class_recalls)) = 0;

    total_true_positives = sum(class_true_positives);
    total_false_positives = sum(class_false_positives);
    total_false_negatives = sum(class_false_negatives);

    total_accuracy = total_true_positives / (total_true_positives + total_false_positives);
    total_precision = total_true_positives / (total_true_positives + total_false_positives);
    total_recall = total_true_positives / (total_true_positives + total_false_negatives);

    class_f1_scores = 2 * (class_precisions .* class_recalls) ./ (class_precisions + class_recalls);
    macro_f1_score = mean(class_f1_scores);

    fprintf('total metrics:\n');
    fprintf('total accuracy: %.2f%%\n', total_accuracy * 100);
    fprintf('total precision: %.2f%%\n', total_precision * 100);
    fprintf('total recall: %.2f%%\n', total_recall * 100);
    fprintf('macro f1-score: %.2f\n', macro_f1_score);
    fprintf('\n');

    for c = 1:num_classes
        fprintf('class %d:\n', c);
        fprintf('accuracy: %.2f%%\n', class_accuracies(c) * 100);
        fprintf('precision: %.2f%%\n', class_precisions(c) * 100);
        fprintf('recall: %.2f%%\n', class_recalls(c) * 100);
        fprintf('f1-score: %.2f\n', class_f1_scores(c));
        fprintf('\n');
    end
end
