from datadiff.filtering import evaluate_filter_predicate, sql_filter_condition


def test_truth_filter_comparator_matches_sql_three_valued_logic():
    assert evaluate_filter_predicate(100, "gt_is_not_true", 150) is True
    assert evaluate_filter_predicate(200, "gt_is_not_true", 150) is False
    assert evaluate_filter_predicate(None, "gt_is_not_true", 150) is True

    assert evaluate_filter_predicate(100, "gt_is_true", 150) is False
    assert evaluate_filter_predicate(200, "gt_is_true", 150) is True
    assert evaluate_filter_predicate(None, "gt_is_true", 150) is False


def test_filter_comparator_treats_nan_as_sql_null():
    nan = float("nan")

    assert evaluate_filter_predicate(nan, "!=", "") is False
    assert evaluate_filter_predicate(nan, "==", nan) is False
    assert evaluate_filter_predicate(nan, "gt_is_not_true", 150) is True
    assert evaluate_filter_predicate(nan, "gt_is_unknown", 150) is True


def test_sql_truth_filter_condition_renders_nested_is_true_form():
    assert (
        sql_filter_condition('q."j"', "150", "gt_is_not_true")
        == 'NOT ((q."j" > 150) IS TRUE)'
    )
    assert sql_filter_condition('q."j"', "150", "gt_is_unknown") == '(q."j" > 150) IS NULL'


def test_set_membership_filter_uses_two_valued_null_safe_semantics():
    assert evaluate_filter_predicate("alpha", "in_set", ["alpha", "中文"]) is True
    assert evaluate_filter_predicate("beta", "in_set", ["alpha", "中文"]) is False
    assert evaluate_filter_predicate(None, "in_set", ["alpha", "中文"]) is False
    assert sql_filter_condition('q."s"', "('alpha', '中文')", "in_set") == 'q."s" IN (\'alpha\', \'中文\')'
