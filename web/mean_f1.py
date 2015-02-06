def true_positives(s1, s2):
    """
    The "true positives" are the intersection of the two lists
      s1: true labels
      s2: predicted labels
    """
    return len(set(s1).intersection(s2))

def false_positives(s1, s2):
    """
    false positives are predicted items that aren't real
      s1: true labels
      s2: predicted labels
    """
    return len(set(s1).difference(s2))

def false_negatives(s1, s2):
    """
    false negatives are real items that aren't predicted
      s1: true labels
      s2: predicted labels
    """
    return len(set(s2).difference(s1))

def precision(s1, s2):
    """
    Precision is the ratio of true positives (tp) to all predicted positives (tp + fp)
      s1: true labels
      s2: predicted labels
    """
    tp = true_positives(s1, s2)
    fp = false_positives(s1, s2)
    if tp == 0 and fp == 0:
        return 0.0
    return 1.0 * tp / (tp + fp)

def recall(s1, s2):
    """
    Recall is the ratio of true positives to all actual positives (tp + fn)
      s1: true labels
      s2: predicted labels
    """
    tp = true_positives(s1, s2)
    fn = false_negatives(s1, s2)
    if tp == 0 and fn == 0:
        return 0.0
    return 1.0 * tp / (tp + fn)

def f1(s1, s2):
    """
    The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision (p) and recall (r).
      s1: true labels
      s2: predicted labels
    """
    p = precision(s1, s2)
    r = recall(s1, s2)
    if p == 0 and r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)

def mean_f1(y_true, y_pred):
    sets = zip(y_true, y_pred)
    return sum([f1(s1, s2) for s1, s2 in sets]) / len(sets)

if __name__ == '__main__':
    y_true = [
              [1, 2],
              [3, 4, 5],
              [6],
              [7]
    ]
    y_pred = [
              [1, 2, 3, 9],
              [3, 4],
              [6, 12],
              [1]
    ]
    print mean_f1(y_true, y_pred)
