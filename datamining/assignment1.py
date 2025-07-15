def PerformanceEvaluator(y_true, y_pred, print_confusion=False):
    if len(y_true) != len(y_pred):
        print("y_true and y_pred have different lengths.")
        return

    # Confusion matrix
    class_number = 10
    confusion_matrix = [[0 for _ in range(class_number)] for _ in range(class_number)]

    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true][pred] += 1

    if print_confusion:
        print("Confusion matrix: ")
        for row in confusion_matrix:
            print(row)

    # Accuracy
    correct = 0
    total = 0

    for i in range(class_number):
        correct += confusion_matrix[i][i]
        total += sum(confusion_matrix[i])
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}({accuracy * 100:.2f}%)")
    
    # macro-Recall, macro-Precision, macro-F1-score
    sum_recall = 0
    sum_precision = 0
    sum_f1_score = 0

    for i in range(class_number):
        tp = confusion_matrix[i][i]
        fn = sum(confusion_matrix[i]) - tp
        fp = sum([confusion_matrix[j][i] for j in range(class_number)]) - tp

        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        sum_recall += recall

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        sum_precision += precision

        if recall + precision == 0:
            f1_score = 0
        else:
            f1_score = 2 * recall * precision / (recall + precision)
        sum_f1_score += f1_score

    print(f"macro-Recall: {sum_recall / class_number:.4f}")
    print(f"macro-Precision: {sum_precision / class_number:.4f}")
    print(f"macro-F1-score: {sum_f1_score / class_number:.4f}")
    