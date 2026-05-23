from datadiff.filtering import evaluate_filter_predicate, sql_filter_condition
from datadiff.tuple_logic import evaluate_tuple_absence


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


def test_range_filter_uses_closed_numeric_bounds():
    assert evaluate_filter_predicate(0, "range_closed", [-1, 1]) is True
    assert evaluate_filter_predicate(-1, "range_closed", [-1, 1]) is True
    assert evaluate_filter_predicate(2, "range_closed", [-1, 1]) is False
    assert evaluate_filter_predicate(None, "range_closed", [-1, 1]) is False
    assert sql_filter_condition('q."x"', "(-1, 1)", "range_closed") == 'q."x" BETWEEN -1 AND 1'


def test_tuple_absence_filter_uses_sql_row_value_not_in_truth_table():
    right_rows = [{"a": 1, "b": 1}, {"a": None, "b": 4}]

    assert evaluate_tuple_absence({"a": 2, "b": 2}, ["a", "b"], right_rows, ["a", "b"]) is True
    assert evaluate_tuple_absence({"a": 1, "b": 1}, ["a", "b"], right_rows, ["a", "b"]) is False
    assert evaluate_tuple_absence({"a": 3, "b": None}, ["a", "b"], right_rows, ["a", "b"]) is False
    assert evaluate_tuple_absence({"a": None, "b": 4}, ["a", "b"], right_rows, ["a", "b"]) is False


def test_null_predicate_filters_are_explicit_null_checks():
    assert evaluate_filter_predicate(None, "is_null", None) is True
    assert evaluate_filter_predicate("alpha", "is_null", None) is False
    assert evaluate_filter_predicate(None, "is_not_null", None) is False
    assert evaluate_filter_predicate("alpha", "is_not_null", None) is True
    assert sql_filter_condition('q."s"', "NULL", "is_null") == 'q."s" IS NULL'
    assert sql_filter_condition('q."s"', "NULL", "is_not_null") == 'q."s" IS NOT NULL'


def test_boolean_predicate_filters_match_sql_truth_tests():
    assert evaluate_filter_predicate(True, "bool_is_true", None) is True
    assert evaluate_filter_predicate(False, "bool_is_true", None) is False
    assert evaluate_filter_predicate(None, "bool_is_true", None) is False

    assert evaluate_filter_predicate(True, "bool_is_not_true", None) is False
    assert evaluate_filter_predicate(False, "bool_is_not_true", None) is True
    assert evaluate_filter_predicate(None, "bool_is_not_true", None) is True

    assert evaluate_filter_predicate(False, "bool_is_false", None) is True
    assert evaluate_filter_predicate(None, "bool_is_not_false", None) is True
    assert evaluate_filter_predicate(None, "bool_is_unknown", None) is True
    assert evaluate_filter_predicate(True, "bool_is_not_unknown", None) is True
    assert sql_filter_condition('q."flag"', "NULL", "bool_is_true") == 'q."flag" IS TRUE'
    assert sql_filter_condition('q."flag"', "NULL", "bool_is_not_true") == 'q."flag" IS NOT TRUE'
    assert sql_filter_condition('q."flag"', "NULL", "bool_is_unknown") == 'q."flag" IS NULL'
