use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyIterator, PyList, PyModule, PyString, PyTuple};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

fn py_number_to_json(payload: &Bound<'_, PyAny>) -> PyResult<serde_json::Number> {
    let text: String = payload.str()?.extract()?;
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Number(number)) => Ok(number),
        Ok(_) => Err(PyValueError::new_err("numeric payload did not convert to a JSON number")),
        Err(err) => Err(PyValueError::new_err(format!(
            "numeric payload is not JSON-serializable: {err}"
        ))),
    }
}

fn py_json_key(payload: &Bound<'_, PyAny>) -> PyResult<String> {
    if payload.is_none() {
        return Ok("null".to_string());
    }
    if payload.downcast::<PyBool>().is_ok() {
        return Ok(if payload.extract::<bool>()? { "true" } else { "false" }.to_string());
    }
    if let Ok(value) = payload.downcast::<PyString>() {
        return value.extract::<String>();
    }
    if payload.downcast::<PyInt>().is_ok() || payload.downcast::<PyFloat>().is_ok() {
        return match py_to_json(payload)? {
            Value::Number(number) => Ok(number.to_string()),
            _ => Err(PyValueError::new_err(
                "mapping keys must be strings, numbers, booleans, or null",
            )),
        };
    }
    Err(PyValueError::new_err(
        "mapping keys must be strings, numbers, booleans, or null",
    ))
}

fn py_to_json(payload: &Bound<'_, PyAny>) -> PyResult<Value> {
    if payload.is_none() {
        return Ok(Value::Null);
    }
    if payload.downcast::<PyBool>().is_ok() {
        return Ok(Value::Bool(payload.extract::<bool>()?));
    }
    if let Ok(value) = payload.downcast::<PyString>() {
        return Ok(Value::String(value.extract::<String>()?));
    }
    if payload.downcast::<PyInt>().is_ok() || payload.downcast::<PyFloat>().is_ok() {
        return Ok(Value::Number(py_number_to_json(payload)?));
    }
    if let Ok(dict) = payload.downcast::<PyDict>() {
        let mut entries: Vec<(String, Value)> = Vec::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            entries.push((py_json_key(&key)?, py_to_json(&value)?));
        }
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let mut object = serde_json::Map::with_capacity(entries.len());
        for (key, value) in entries {
            object.insert(key, value);
        }
        return Ok(Value::Object(object));
    }
    if let Ok(list) = payload.downcast::<PyList>() {
        let mut out: Vec<Value> = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(Value::Array(out));
    }
    if let Ok(tuple) = payload.downcast::<PyTuple>() {
        let mut out: Vec<Value> = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            out.push(py_to_json(&item)?);
        }
        return Ok(Value::Array(out));
    }
    Err(PyValueError::new_err(
        "payload is not JSON-serializable by rust_kernel",
    ))
}

fn json_string(value: &Value) -> PyResult<String> {
    serde_json::to_string(value).map_err(|err| PyValueError::new_err(err.to_string()))
}

fn row_keys_from_array(items: Vec<Value>, error_context: &str) -> PyResult<Vec<String>> {
    items
        .into_iter()
        .map(|item| serde_json::to_string(&item).map_err(|err| PyValueError::new_err(err.to_string())))
        .collect::<PyResult<Vec<String>>>()
        .map_err(|err| PyValueError::new_err(format!("{error_context}: {err}")))
}

fn row_keys_from_value(value: Value, error_context: &str) -> PyResult<Vec<String>> {
    let Value::Array(items) = value else {
        return Err(PyValueError::new_err(error_context.to_string()));
    };
    row_keys_from_array(items, error_context)
}

fn sorted_key_order(keys: &[String]) -> (Vec<usize>, Vec<String>) {
    let mut indexed: Vec<(usize, String)> = keys
        .iter()
        .enumerate()
        .map(|(idx, key)| (idx, key.clone()))
        .collect();
    indexed.sort_by(|left, right| left.1.cmp(&right.1));
    let indices = indexed.iter().map(|(idx, _)| *idx).collect();
    let ordered_keys = indexed.into_iter().map(|(_, key)| key).collect();
    (indices, ordered_keys)
}

fn ordered_key_signature(keys: &[String]) -> PyResult<String> {
    serde_json::to_string(keys).map_err(|err| PyValueError::new_err(err.to_string()))
}

fn has_duplicate_key_runs(keys: &[String]) -> bool {
    keys.windows(2).any(|window| window[0] == window[1])
}

fn row_profile_from_keys(stable: &[String]) -> PyResult<(String, String, bool)> {
    let ordered = ordered_key_signature(stable)?;
    let (_, sorted_keys) = sorted_key_order(stable);
    let unordered = ordered_key_signature(&sorted_keys)?;
    let has_duplicates = has_duplicate_key_runs(&sorted_keys);
    Ok((ordered, unordered, has_duplicates))
}

fn multiset_diff_from_keys(left: &[String], right: &[String]) -> (Vec<String>, Vec<String>) {
    let mut left_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut right_counts: BTreeMap<String, usize> = BTreeMap::new();

    for row in left {
        *left_counts.entry(row.clone()).or_insert(0usize) += 1;
    }
    for row in right {
        *right_counts.entry(row.clone()).or_insert(0usize) += 1;
    }

    let mut left_only: Vec<String> = Vec::new();
    let mut right_only: Vec<String> = Vec::new();

    for (row, count) in &left_counts {
        let matched = right_counts.get(row).copied().unwrap_or(0usize);
        for _ in 0..count.saturating_sub(matched) {
            left_only.push(row.clone());
        }
    }
    for (row, count) in &right_counts {
        let matched = left_counts.get(row).copied().unwrap_or(0usize);
        for _ in 0..count.saturating_sub(matched) {
            right_only.push(row.clone());
        }
    }
    (left_only, right_only)
}

fn ordered_group_ids_from_signatures(signatures: &[String]) -> Vec<usize> {
    let mut group_ids: Vec<usize> = Vec::with_capacity(signatures.len());
    let mut groups: BTreeMap<String, usize> = BTreeMap::new();
    for signature in signatures {
        let next_group = groups.len();
        let group_id = *groups.entry(signature.clone()).or_insert(next_group);
        group_ids.push(group_id);
    }
    group_ids
}

fn grouped_indices_from_group_ids(group_ids: &[usize]) -> Vec<Vec<usize>> {
    let mut grouped: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, group_id) in group_ids.iter().enumerate() {
        grouped.entry(*group_id).or_default().push(index);
    }
    grouped.into_values().collect()
}

fn majority_group_from_group_ids(group_ids: &[usize]) -> Option<Vec<usize>> {
    if group_ids.len() < 2 {
        return None;
    }
    let groups = grouped_indices_from_group_ids(group_ids);
    if groups.len() < 2 {
        return None;
    }
    let max_size = groups.iter().map(|group| group.len()).max().unwrap_or(0usize);
    if groups
        .iter()
        .filter(|group| group.len() == max_size)
        .count()
        != 1
    {
        return None;
    }
    groups.into_iter().max_by_key(|group| group.len())
}

fn suspicious_indices_from_group_ids(group_ids: &[usize]) -> Vec<usize> {
    let groups = grouped_indices_from_group_ids(group_ids);
    if groups.len() < 2 {
        return Vec::new();
    }
    let Some(majority_group) = majority_group_from_group_ids(group_ids) else {
        return (0..group_ids.len()).collect();
    };
    let majority_lookup: BTreeSet<usize> = majority_group.iter().copied().collect();
    (0..group_ids.len())
        .filter(|index| !majority_lookup.contains(index))
        .collect()
}

fn anchor_matching_indices_from_group_ids(group_ids: &[usize], anchor_index: usize) -> PyResult<Vec<usize>> {
    if anchor_index >= group_ids.len() {
        return Err(PyValueError::new_err("anchor_index out of range"));
    }
    let anchor_group = group_ids[anchor_index];
    Ok(group_ids
        .iter()
        .enumerate()
        .filter_map(|(index, group_id)| (*group_id == anchor_group).then_some(index))
        .collect())
}

fn anchor_mismatching_indices_from_group_ids(group_ids: &[usize], anchor_index: usize) -> PyResult<Vec<usize>> {
    if anchor_index >= group_ids.len() {
        return Err(PyValueError::new_err("anchor_index out of range"));
    }
    let anchor_group = group_ids[anchor_index];
    Ok(group_ids
        .iter()
        .enumerate()
        .filter_map(|(index, group_id)| (*group_id != anchor_group).then_some(index))
        .collect())
}

fn anchor_confidence_from_group_ids(group_ids: &[usize], anchor_index: usize) -> PyResult<String> {
    let mismatching = anchor_mismatching_indices_from_group_ids(group_ids, anchor_index)?;
    if mismatching.is_empty() {
        return Ok("high".to_string());
    }
    Ok(
        if mismatching.len() < std::cmp::max(1usize, group_ids.len().saturating_sub(1)) {
            "high"
        } else {
            "medium"
        }
        .to_string(),
    )
}

fn result_comparison_key(status: &str, columns: &[String], ordered_row_signature: &str, error_type: &str) -> PyResult<String> {
    json_string(&serde_json::json!({
        "status": status,
        "columns": columns,
        "row_signature": ordered_row_signature,
        "error_type": error_type,
    }))
}

struct RowSetProfileRecord {
    ordered_signature: String,
    unordered_signature: String,
    has_duplicates: bool,
    row_count: usize,
}

fn row_set_profile_record_from_rows(rows: Vec<Value>, error_context: &str) -> PyResult<RowSetProfileRecord> {
    let row_count = rows.len();
    let row_keys = row_keys_from_array(rows, error_context)?;
    let (ordered_signature, unordered_signature, has_duplicates) = row_profile_from_keys(&row_keys)?;
    Ok(RowSetProfileRecord {
        ordered_signature,
        unordered_signature,
        has_duplicates,
        row_count,
    })
}

fn parse_row_set_profiles(value: Value, error_context: &str) -> PyResult<Vec<RowSetProfileRecord>> {
    let Value::Array(items) = value else {
        return Err(PyValueError::new_err(error_context.to_string()));
    };
    let mut profiles: Vec<RowSetProfileRecord> = Vec::with_capacity(items.len());

    for item in items {
        let Value::Array(rows) = item else {
            return Err(PyValueError::new_err("each row set must be a JSON array"));
        };
        profiles.push(row_set_profile_record_from_rows(
            rows,
            "each row set must be a JSON array",
        )?);
    }

    Ok(profiles)
}

fn ordered_signatures_from_row_profiles(profiles: &[RowSetProfileRecord]) -> Vec<String> {
    profiles
        .iter()
        .map(|profile| profile.ordered_signature.clone())
        .collect()
}

fn row_group_ids_from_profiles(profiles: &[RowSetProfileRecord]) -> Vec<usize> {
    ordered_group_ids_from_signatures(&ordered_signatures_from_row_profiles(profiles))
}

fn row_mismatch_class_from_profiles(profiles: &[RowSetProfileRecord]) -> String {
    if profiles.is_empty() {
        return "none".to_string();
    }
    let first_row_count = profiles[0].row_count;
    if profiles.iter().any(|profile| profile.row_count != first_row_count) {
        return "row_count".to_string();
    }
    let first_ordered = &profiles[0].ordered_signature;
    if profiles
        .iter()
        .all(|profile| &profile.ordered_signature == first_ordered)
    {
        return "none".to_string();
    }
    let first_unordered = &profiles[0].unordered_signature;
    if profiles
        .iter()
        .all(|profile| &profile.unordered_signature == first_unordered)
    {
        return "row_order".to_string();
    }
    "value".to_string()
}

struct ResultProfileRecord {
    status: String,
    columns: Vec<String>,
    error_type: String,
    row_profile: RowSetProfileRecord,
    comparison_key: String,
}

fn parse_result_profiles(value: Value) -> PyResult<Vec<ResultProfileRecord>> {
    let Value::Array(items) = value else {
        return Err(PyValueError::new_err("results must be a JSON array"));
    };
    let mut profiles: Vec<ResultProfileRecord> = Vec::with_capacity(items.len());

    for item in items {
        let Value::Object(object) = item else {
            return Err(PyValueError::new_err("each result must be a JSON object"));
        };
        let status = object
            .get("status")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        let error_type = object
            .get("error_type")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .to_string();
        let columns = match object.get("columns") {
            Some(Value::Array(values)) => values
                .iter()
                .map(|value| value.as_str().unwrap_or("").to_string())
                .collect::<Vec<String>>(),
            Some(_) => return Err(PyValueError::new_err("result columns must be a JSON array")),
            None => Vec::new(),
        };
        let rows = match object.get("rows") {
            Some(Value::Array(values)) => values.clone(),
            Some(_) => return Err(PyValueError::new_err("result rows must be a JSON array")),
            None => Vec::new(),
        };
        let row_profile = row_set_profile_record_from_rows(rows, "result rows must be a JSON array")?;
        let comparison_key = result_comparison_key(&status, &columns, &row_profile.ordered_signature, &error_type)?;
        profiles.push(ResultProfileRecord {
            status,
            columns,
            error_type,
            row_profile,
            comparison_key,
        });
    }

    Ok(profiles)
}

#[pyfunction]
fn json_canonical_dumps(py: Python<'_>, payload: PyObject) -> PyResult<String> {
    let value = py_to_json(payload.bind(py))?;
    json_string(&value)
}

#[pyfunction]
fn short_sha256_hex(py: Python<'_>, payload: PyObject, length: usize) -> PyResult<String> {
    let canonical = json_canonical_dumps(py, payload)?;
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let full = format!("{:x}", hasher.finalize());
    Ok(full.chars().take(length).collect())
}

#[pyfunction]
fn canonical_sort_key(py: Python<'_>, payload: PyObject) -> PyResult<String> {
    json_canonical_dumps(py, payload)
}

#[pyfunction]
fn canonical_group_keys(py: Python<'_>, payloads: PyObject) -> PyResult<Vec<String>> {
    row_keys_from_value(py_to_json(payloads.bind(py))?, "payloads must be a JSON array")
}

#[pyfunction]
fn stable_rows(py: Python<'_>, rows: PyObject) -> PyResult<Vec<String>> {
    row_keys_from_value(py_to_json(rows.bind(py))?, "rows must be a JSON array")
}

fn feature_bucket(prefix: &str, value: i64, limits: &[(i64, &str)], fallback: &str) -> String {
    for (limit, name) in limits {
        if value <= *limit {
            return format!("{prefix}:{name}");
        }
    }
    format!("{prefix}:{fallback}")
}

fn json_object_str(object: &serde_json::Map<String, Value>, key: &str, default: &str) -> String {
    object
        .get(key)
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .unwrap_or(default)
        .to_string()
}

fn json_object_i64(object: &serde_json::Map<String, Value>, key: &str, default: i64) -> i64 {
    object
        .get(key)
        .and_then(|value| value.as_i64().or_else(|| value.as_u64().and_then(|item| i64::try_from(item).ok())))
        .unwrap_or(default)
}

fn json_object_bool(object: &serde_json::Map<String, Value>, key: &str, default: bool) -> Option<bool> {
    match object.get(key) {
        None => Some(default),
        Some(Value::Bool(value)) => Some(*value),
        _ => None,
    }
}

fn json_string_list(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str().map(str::to_string))
            .filter(|item| !item.is_empty())
            .collect(),
        Some(Value::String(item)) if !item.is_empty() => vec![item.clone()],
        _ => Vec::new(),
    }
}

fn operation_kind(object: &serde_json::Map<String, Value>) -> String {
    json_object_str(object, "op", "unknown")
}

fn operation_columns(object: &serde_json::Map<String, Value>) -> Vec<String> {
    json_string_list(object.get("columns"))
}

fn operation_values_len(object: &serde_json::Map<String, Value>) -> i64 {
    match object.get("values") {
        Some(Value::Array(values)) => values.len() as i64,
        Some(Value::Null) | None => 0,
        Some(Value::String(value)) if value.is_empty() => 0,
        Some(_) => 1,
    }
}

fn operation_key_list(object: &serde_json::Map<String, Value>, list_key: &str, scalar_key: &str) -> Vec<String> {
    let list = json_string_list(object.get(list_key));
    if !list.is_empty() {
        return list;
    }
    json_string_list(object.get(scalar_key))
}

fn expression_payload(object: &serde_json::Map<String, Value>) -> Option<&serde_json::Map<String, Value>> {
    object.get("expr").and_then(|value| value.as_object())
}

fn expression_str(object: &serde_json::Map<String, Value>, key: &str, default: &str) -> String {
    expression_payload(object)
        .map(|expr| json_object_str(expr, key, default))
        .unwrap_or_else(|| default.to_string())
}

fn literal_feature_type(values: &[&Value]) -> String {
    let mut types: BTreeSet<&'static str> = BTreeSet::new();
    for value in values {
        match value {
            Value::Null => {
                types.insert("null");
            }
            Value::Bool(_) => {
                types.insert("bool");
            }
            Value::Number(number) => {
                if number.is_i64() || number.is_u64() {
                    types.insert("int");
                } else {
                    types.insert("float");
                }
            }
            Value::String(_) => {
                types.insert("str");
            }
            _ => {
                types.insert("other");
            }
        }
    }
    if types.len() == 1 {
        return types.iter().next().unwrap_or(&"other").to_string();
    }
    if types.iter().all(|item| *item == "int" || *item == "float") {
        return "numeric".to_string();
    }
    "mixed".to_string()
}

fn sort_direction_feature(object: &serde_json::Map<String, Value>) -> String {
    let mut directions: BTreeSet<bool> = BTreeSet::new();
    if let Some(Value::Array(keys)) = object.get("keys") {
        for key in keys {
            let Some(key_object) = key.as_object() else {
                continue;
            };
            match json_object_bool(key_object, "ascending", true) {
                Some(value) => {
                    directions.insert(value);
                }
                None => return "sort:asc".to_string(),
            }
        }
    } else if let Some(ascending) = json_object_bool(object, "ascending", true) {
        if !operation_columns(object).is_empty() {
            directions.insert(ascending);
        }
    } else {
        return "sort:asc".to_string();
    }
    if directions.len() > 1 {
        return "sort:mixed".to_string();
    }
    if directions.iter().next().copied().unwrap_or(true) {
        "sort:asc".to_string()
    } else {
        "sort:desc".to_string()
    }
}

fn sort_key_count(object: &serde_json::Map<String, Value>, key: &str) -> i64 {
    match object.get(key) {
        Some(Value::Array(values)) => values.len() as i64,
        _ => 0,
    }
}

fn add_filter_comparator_features(features: &mut BTreeSet<String>, cmp: &str) {
    if cmp == "in_set" || cmp == "not_in_set" {
        features.insert("filter:set-membership".to_string());
        if cmp == "not_in_set" {
            features.insert("filter:negative-set-membership".to_string());
        }
    }
    if cmp == "is_null" || cmp == "is_not_null" {
        features.insert("filter:null-predicate".to_string());
        features.insert(format!("filter:null-predicate:{cmp}"));
    }
    if let Some(truth) = cmp.strip_prefix("bool_") {
        features.insert("filter:boolean-predicate".to_string());
        features.insert(format!("filter:boolean-predicate:{truth}"));
    }
    if cmp == "range_closed" {
        features.insert("filter:range-closed".to_string());
    }
    if matches!(cmp, "str_contains" | "str_starts_with" | "str_ends_with") {
        let suffix = cmp.replace("str_", "string-").replace('_', "-");
        features.insert("filter:string-pattern".to_string());
        features.insert(format!("filter:{suffix}"));
    }
    for token in ["gt", "ge", "lt", "le", "eq", "ne"] {
        let prefix = format!("{token}_");
        if let Some(truth) = cmp.strip_prefix(&prefix) {
            if matches!(
                truth,
                "is_true" | "is_not_true" | "is_false" | "is_not_false" | "is_unknown" | "is_not_unknown"
            ) {
                features.insert("filter:truth-test".to_string());
                features.insert(format!("filter:truth:{truth}"));
            }
        }
    }
}

fn add_probe_features(features: &mut BTreeSet<String>, kind: &str, object: &serde_json::Map<String, Value>) -> bool {
    let pairs: &[(&str, &[&str])] = &[
        ("scalar_subquery_probe", &["subquery:correlated-scalar", "subquery:nested-aggregate"]),
        ("window_avg_probe", &["window:rows-frame", "window:avg"]),
        ("struct_distinct_probe", &["struct:unnest", "struct:distinct"]),
        ("bit_compare_probe", &["bit:unequal-length", "comparison:bit-order"]),
        ("round_even_probe", &["numeric:round-even", "float:decimal-scale"]),
        ("float_literal_precision_probe", &["duckdb:float-literal-precision", "float:literal-cast-consistency", "float:decimal-literal"]),
        ("timestamp_precision_filter_probe", &["polars:timestamp-precision-filter", "timestamp:precision-filter", "timestamp:unit-cast"]),
        ("series_rtruediv_probe", &["series:reverse-division", "arithmetic:operand-order"]),
        ("uint64_isin_probe", &["pandas:uint64-isin", "membership:unsigned-precision"]),
        ("tuple_anti_null_probe", &["duckdb:tuple-anti-null", "nulls:ternary-membership"]),
        ("setop_all_duplicate_probe", &["datafusion:setop-all-duplicate-count", "sql:setop-all", "setop:except-all", "setop:intersect-all", "setop:duplicate-count"]),
        ("json_predicate_order_probe", &["duckdb:json-predicate-order", "json:predicate-reorder"]),
        ("sparse_mask_probe", &["pandas:sparse-mask", "mask:sparse-array"]),
        ("float_wrap_probe", &["polars:wrap-numerical", "cast:float-overflow"]),
        ("index_bool_probe", &["pandas:index-bool", "api:result-type"]),
        ("empty_literal_groupby_probe", &["polars:empty-literal-groupby", "groupby:empty-literal"]),
        ("arrow_string_eq_sum_probe", &["pandas:arrow-string-eq-sum", "arrow:string-bool-reduction"]),
        ("arrow_timestamp_loc_slice_probe", &["pandas:arrow-timestamp-loc-slice", "arrow:timestamp-index-slice"]),
        ("arrow_timestamp_index_attr_probe", &["pandas:arrow-timestamp-index-attr", "arrow:timestamp-index-attribute"]),
        ("eval_inplace_alias_probe", &["pandas:eval-inplace-alias", "copy:on-write-alias"]),
        ("bool_reduction_skipna_probe", &["pandas:bool-reduction-skipna", "nullable-bool:reduction"]),
        ("dataset_isin_all_match_probe", &["pyarrow:dataset-isin-all-match", "dataset:membership-filter"]),
        ("run_end_null_compute_probe", &["pyarrow:run-end-null-compute", "run_end:null-compute"]),
        ("large_string_partition_probe", &["pyarrow:large-string-partition", "dataset:partition-schema"]),
        ("hash_pivot_wider_probe", &["pyarrow:hash-pivot-wider", "pivot:wider-order"]),
        ("list_flatten_parent_indices_probe", &["pyarrow:list-flatten-parent-indices", "arrow:list-layout"]),
        ("rolling_mean_by_null_count_probe", &["polars:rolling-mean-by-null-count", "rolling:temporal-min-samples"]),
    ];
    if kind == "random_case_probe" {
        features.insert("case_expr:simple".to_string());
        features.insert("case_expr:random-subject".to_string());
        features.insert(feature_bucket("case_probe_rows", json_object_i64(object, "rows", 0), &[(1000, "small"), (10000, "medium")], "large"));
        return true;
    }
    if kind == "group_quantile_probe" {
        features.insert("quantile:dynamic-key".to_string());
        features.insert(feature_bucket("quantile_probe_values", operation_values_len(object), &[(2, "tiny"), (4, "small")], "medium"));
        return true;
    }
    if kind == "csv_long_numeric_roundtrip_probe" {
        features.insert("csv:long-numeric-roundtrip".to_string());
        features.insert("csv:numeric-inference".to_string());
        features.insert("numeric:long-identifier".to_string());
        features.insert(feature_bucket("csv_probe_values", operation_values_len(object), &[(3, "few"), (6, "several")], "many"));
        return true;
    }
    for (probe_kind, probe_features) in pairs {
        if *probe_kind == kind {
            for feature in *probe_features {
                features.insert((*feature).to_string());
            }
            return true;
        }
    }
    false
}

#[pyfunction]
fn extract_case_features(
    py: Python<'_>,
    operations: PyObject,
    column_types: PyObject,
) -> PyResult<Vec<String>> {
    let operations_value = py_to_json(operations.bind(py))?;
    let column_types_value = py_to_json(column_types.bind(py))?;
    let Value::Array(operation_values) = operations_value else {
        return Err(PyValueError::new_err("operations must be a JSON array"));
    };
    let column_types = match column_types_value {
        Value::Object(object) => object
            .into_iter()
            .filter_map(|(key, value)| value.as_str().map(|text| (key, text.to_string())))
            .collect::<BTreeMap<String, String>>(),
        _ => return Err(PyValueError::new_err("column_types must be a JSON object")),
    };
    let mut features: BTreeSet<String> = BTreeSet::new();
    let mut op_names: Vec<String> = Vec::with_capacity(operation_values.len());

    for value in &operation_values {
        let Some(object) = value.as_object() else {
            continue;
        };
        let kind = operation_kind(object);
        op_names.push(kind.clone());
        features.insert(format!("op:{kind}"));

        match kind.as_str() {
            "filter" => {
                let cmp = json_object_str(object, "cmp", "unknown");
                let column = json_object_str(object, "column", "unknown");
                features.insert(format!("cmp:{cmp}"));
                if let Some(column_type) = column_types.get(&column) {
                    features.insert(format!("filter_type:{column_type}"));
                }
                add_filter_comparator_features(&mut features, &cmp);
            }
            "tuple_absence_filter" => {
                features.insert("filter:tuple-absence".to_string());
            }
            "union_all" => {
                features.insert("table:row-append".to_string());
                features.insert("union_all:append".to_string());
            }
            "drop_nulls" => {
                let width = operation_columns(object).len() as i64;
                features.insert("null:drop".to_string());
                features.insert("drop_nulls:subset".to_string());
                features.insert(feature_bucket("drop_nulls_columns", width, &[(1, "one"), (2, "two")], "many"));
            }
            "semi_join" | "anti_join" => {
                let left_keys = operation_key_list(object, "left_keys", "left_on");
                let right_keys = operation_key_list(object, "right_keys", "right_on");
                features.insert("join:existence".to_string());
                features.insert(if kind == "semi_join" { "membership:semi_join" } else { "membership:anti_join" }.to_string());
                features.insert(format!("membership:left_key:{}", if left_keys.is_empty() { "unknown".to_string() } else { left_keys.join(",") }));
                features.insert(format!("membership:right_key:{}", if right_keys.is_empty() { "unknown".to_string() } else { right_keys.join(",") }));
            }
            "distinct" => {
                let width = operation_columns(object).len() as i64;
                features.insert("distinct:deduplicate".to_string());
                features.insert(feature_bucket("distinct_columns", width, &[(1, "one"), (2, "two")], "many"));
            }
            "fill_null" => {
                let column = json_object_str(object, "column", "");
                features.insert("null:fill".to_string());
                if let Some(column_type) = column_types.get(&column) {
                    features.insert(format!("fill_null_type:{column_type}"));
                }
                match object.get("value") {
                    Some(Value::Bool(false)) => {
                        features.insert("fill_null:false".to_string());
                    }
                    Some(Value::String(value)) if value.is_empty() => {
                        features.insert("fill_null:empty-string".to_string());
                    }
                    Some(Value::Number(number)) if number.as_i64() == Some(0) || number.as_u64() == Some(0) => {
                        features.insert("fill_null:zero".to_string());
                    }
                    _ => {}
                }
            }
            "coalesce" => {
                let columns = operation_columns(object);
                let alias = json_object_str(object, "as", "");
                features.insert("null:coalesce".to_string());
                features.insert("coalesce:columns".to_string());
                features.insert(feature_bucket("coalesce_columns", columns.len() as i64, &[(2, "two"), (3, "three")], "many"));
                if let Some(first_column) = columns.first() {
                    if let Some(column_type) = column_types.get(first_column) {
                        features.insert(format!("coalesce_type:{column_type}"));
                    }
                }
                if column_types.contains_key(&alias) {
                    features.insert("coalesce:overwrite".to_string());
                }
                if object.contains_key("fallback") {
                    features.insert("coalesce:fallback".to_string());
                }
            }
            "case_when" => {
                let condition = object.get("condition").and_then(|value| value.as_object());
                let column = condition
                    .map(|item| json_object_str(item, "column", ""))
                    .filter(|item| !item.is_empty())
                    .unwrap_or_else(|| json_object_str(object, "column", ""));
                let cmp = condition
                    .map(|item| json_object_str(item, "cmp", "unknown"))
                    .unwrap_or_else(|| json_object_str(object, "cmp", "unknown"));
                features.insert("conditional:case_when".to_string());
                if let Some(column_type) = column_types.get(&column) {
                    features.insert(format!("case_when_type:{column_type}"));
                }
                features.insert(format!("case_when_cmp:{cmp}"));
                let mut values = Vec::new();
                if let Some(value) = object.get("then") {
                    values.push(value);
                }
                if let Some(value) = object.get("else") {
                    values.push(value);
                }
                features.insert(format!("case_when_output:{}", literal_feature_type(&values)));
            }
            "row_number_filter" => {
                let partition_count = json_string_list(object.get("partition_by")).len() as i64;
                let order_count = sort_key_count(object, "order_by");
                features.insert("row_pick:keyed".to_string());
                features.insert(format!("row_pick:cmp:{}", json_object_str(object, "cmp", "unknown")));
                features.insert(feature_bucket("row_pick_partition_count", partition_count, &[(0, "none"), (1, "one")], "many"));
                features.insert(feature_bucket("row_pick_order_count", order_count, &[(1, "one"), (2, "two")], "many"));
            }
            "running_sum" => {
                let input_dtype = json_object_str(object, "input_dtype", "float64");
                let source = json_object_str(object, "source", "");
                features.insert(format!("running:{input_dtype}"));
                if let Some(column_type) = column_types.get(&source) {
                    features.insert(format!("running_source_type:{column_type}"));
                }
                if !json_string_list(object.get("partition_by")).is_empty() {
                    features.insert("running:partitioned".to_string());
                }
            }
            "sortedness_check" => {
                let column = json_object_str(object, "column", "");
                features.insert(format!("sortedness:nulls:{}", json_object_str(object, "nulls", "last")));
                features.insert(if json_object_bool(object, "ascending", true).unwrap_or(true) { "sortedness:asc" } else { "sortedness:desc" }.to_string());
                if let Some(column_type) = column_types.get(&column) {
                    features.insert(format!("sortedness_source_type:{column_type}"));
                }
            }
            "select" => {
                let width = operation_columns(object).len() as i64;
                features.insert(feature_bucket("select_width", width, &[(1, "one"), (3, "few")], "many"));
            }
            "sort" => {
                features.insert(sort_direction_feature(object));
            }
            "join" => {
                features.insert(format!("join:{}", json_object_str(object, "how", "unknown")));
                features.insert(format!("join_table:{}", json_object_str(object, "table", "unknown")));
            }
            "limit" => {
                let limit = json_object_i64(object, "n", 0);
                if limit == 0 {
                    features.insert("op:limit_zero".to_string());
                }
                features.insert(feature_bucket("limit", limit, &[(0, "zero"), (3, "tiny"), (10, "small")], "large"));
            }
            "offset" => {
                let offset = json_object_i64(object, "n", 0);
                if offset == 0 {
                    features.insert("op:offset_zero".to_string());
                }
                features.insert(feature_bucket("offset", offset, &[(0, "zero"), (3, "tiny"), (10, "small")], "large"));
            }
            "mutate" => {
                let mutate_kind = expression_str(object, "kind", "unknown");
                let source = expression_str(object, "source", "");
                features.insert(format!("mutate:{mutate_kind}"));
                features.insert(format!("expr:{mutate_kind}"));
                match mutate_kind.as_str() {
                    "arith_const" => {
                        features.insert(format!("arith:{}", expression_str(object, "op", "unknown")));
                    }
                    "reverse_division_columns" => {
                        features.insert("arithmetic:operand-order".to_string());
                        features.insert("arithmetic:reverse-division".to_string());
                    }
                    "abs" => {
                        features.insert("numeric:abs".to_string());
                    }
                    "clip" => {
                        features.insert("numeric:clip".to_string());
                    }
                    "bool_not" => {
                        features.insert("boolean:not".to_string());
                    }
                    "cast" => {
                        let target = expression_str(object, "to", "unknown");
                        features.insert(format!("cast_to:{target}"));
                        if let Some(source_type) = column_types.get(&source) {
                            features.insert(format!("cast:{source_type}_to_{target}"));
                        }
                        let input_domain = expression_str(object, "input_domain", "");
                        if !input_domain.is_empty() {
                            features.insert(format!("cast_domain:{input_domain}"));
                        }
                    }
                    "string_length" => {
                        features.insert("string:length".to_string());
                    }
                    "string_lower" => {
                        features.insert("string:lower".to_string());
                    }
                    "string_upper" => {
                        features.insert("string:upper".to_string());
                    }
                    "string_strip" => {
                        features.insert("string:strip".to_string());
                    }
                    "string_null_if_empty" => {
                        features.insert("string:null-if-empty".to_string());
                        features.insert("null:empty-string".to_string());
                    }
                    "string_replace" => {
                        features.insert("string:replace".to_string());
                    }
                    "string_slice" => {
                        features.insert("string:slice".to_string());
                    }
                    "string_split_part" => {
                        features.insert("string:split-first".to_string());
                    }
                    "string_concat" => {
                        features.insert("string:concat".to_string());
                    }
                    "string_contains" => {
                        features.insert("string:contains".to_string());
                    }
                    "string_starts_with" => {
                        features.insert("string:starts-with".to_string());
                    }
                    "string_ends_with" => {
                        features.insert("string:ends-with".to_string());
                    }
                    "date_part" => {
                        features.insert("date:part".to_string());
                        features.insert(format!("date_part:{}", expression_str(object, "part", "unknown")));
                    }
                    "string_basename" => {
                        features.insert("path:basename".to_string());
                    }
                    _ => {}
                }
            }
            "groupby" => {
                let keys = json_string_list(object.get("keys"));
                let aggs = object.get("aggs").and_then(|value| value.as_array()).cloned().unwrap_or_default();
                let mut funcs: BTreeSet<String> = BTreeSet::new();
                for agg in &aggs {
                    if let Some(agg_object) = agg.as_object() {
                        funcs.insert(json_object_str(agg_object, "func", "unknown"));
                    }
                }
                if !funcs.is_empty() && funcs.iter().all(|func| matches!(func.as_str(), "count" | "nunique" | "min" | "max" | "any" | "all")) {
                    features.insert("groupby:exact-aggregate".to_string());
                    features.insert("groupby:sorted-input".to_string());
                }
                features.insert(feature_bucket("groupby_keys", keys.len() as i64, &[(1, "one"), (2, "two")], "many"));
                if keys.len() > 1 {
                    features.insert("groupby:multi-key".to_string());
                }
                for key in &keys {
                    if let Some(column_type) = column_types.get(key) {
                        features.insert(format!("group_key_type:{column_type}"));
                    }
                }
                add_aggregate_features(&mut features, &aggs, &column_types);
            }
            "aggregate" => {
                let aggs = object.get("aggs").and_then(|value| value.as_array()).cloned().unwrap_or_default();
                add_aggregate_features(&mut features, &aggs, &column_types);
            }
            _ => {
                add_probe_features(&mut features, &kind, object);
            }
        }
        if !matches!(
            kind.as_str(),
            "random_case_probe" | "group_quantile_probe" | "csv_long_numeric_roundtrip_probe"
        ) {
            add_probe_features(&mut features, &kind, object);
        }
    }

    if !op_names.is_empty() {
        features.insert(format!("opseq:{}", op_names.join(">")));
        features.insert(feature_bucket("op_count", op_names.len() as i64, &[(1, "one"), (3, "few"), (5, "many")], "deep"));
    }

    Ok(features.into_iter().collect())
}

fn add_aggregate_features(
    features: &mut BTreeSet<String>,
    aggs: &[Value],
    column_types: &BTreeMap<String, String>,
) {
    for agg in aggs {
        let Some(agg_object) = agg.as_object() else {
            continue;
        };
        let source = json_object_str(agg_object, "column", "");
        let func = json_object_str(agg_object, "func", "unknown");
        features.insert(format!("agg:{func}"));
        if let Some(source_type) = column_types.get(&source) {
            features.insert(format!("agg_source_type:{source_type}"));
            if source_type == "float" && matches!(func.as_str(), "sum" | "mean") {
                features.insert("agg:precision-float".to_string());
            }
            if source_type == "bool" {
                features.insert("agg:boolean".to_string());
                features.insert(format!("agg:{func}:bool"));
            }
            if func == "count" && source_type == "str" {
                features.insert("agg:count:str".to_string());
            }
            if func == "nunique" {
                features.insert(format!("agg:nunique:{source_type}"));
            }
        }
    }
}

#[pyfunction]
fn compute_minhash(py: Python<'_>, tokens: PyObject, signature_size: usize) -> PyResult<Vec<u64>> {
    let size = std::cmp::max(1usize, signature_size);
    if size > (u16::MAX as usize) + 1 {
        return Err(PyValueError::new_err("signature_size exceeds 16-bit seed space"));
    }
    let mut unique: BTreeSet<String> = BTreeSet::new();
    let bound = tokens.bind(py);
    let iterator = PyIterator::from_bound_object(bound)?;
    for item in iterator {
        let token: String = item?.str()?.extract()?;
        if !token.is_empty() {
            unique.insert(token);
        }
    }
    if unique.is_empty() {
        return Ok(vec![0u64; size]);
    }
    let mut signature = vec![u64::MAX; size];
    for token in unique {
        let encoded = token.as_bytes();
        for index in 0..size {
            let mut hasher = Sha256::new();
            hasher.update((index as u16).to_be_bytes());
            hasher.update([0u8]);
            hasher.update(encoded);
            let digest = hasher.finalize();
            let value = u64::from_be_bytes([
                digest[0], digest[1], digest[2], digest[3],
                digest[4], digest[5], digest[6], digest[7],
            ]);
            if value < signature[index] {
                signature[index] = value;
            }
        }
    }
    Ok(signature)
}

fn counts_map_from_py(payload: &Bound<'_, PyAny>, label: &str) -> PyResult<BTreeMap<String, f64>> {
    let dict = payload
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err(format!("{label} must be a mapping")))?;
    let mut out: BTreeMap<String, f64> = BTreeMap::new();
    for (key, value) in dict.iter() {
        let feature: String = key.str()?.extract()?;
        let count = value.extract::<f64>().unwrap_or(0.0);
        out.insert(feature, count);
    }
    Ok(out)
}

fn bounded_finding_signal(count: f64) -> f64 {
    if count <= 0.0 {
        return 0.0;
    }
    count.ln_1p().min(2.0) / (1.0 + count / 50.0)
}

fn feature_saturation(count: f64) -> f64 {
    if count <= 25.0 {
        return 0.0;
    }
    (count - 25.0).ln_1p()
}

fn profile_saturation(count: f64) -> f64 {
    if count <= 3.0 {
        return 0.0;
    }
    ((count - 3.0).ln_1p() * 1.15).min(4.0)
}

#[pyfunction]
fn score_candidate_feature_metrics_batch(
    py: Python<'_>,
    candidate_specs: PyObject,
    feature_counts: PyObject,
    finding_feature_counts: PyObject,
) -> PyResult<Vec<(f64, f64, f64, f64, f64, usize, usize, f64, f64)>> {
    let feature_counts = counts_map_from_py(feature_counts.bind(py), "feature_counts")?;
    let finding_feature_counts = counts_map_from_py(
        finding_feature_counts.bind(py),
        "finding_feature_counts",
    )?;
    let candidates = PyIterator::from_bound_object(candidate_specs.bind(py))?;
    let mut out: Vec<(f64, f64, f64, f64, f64, usize, usize, f64, f64)> = Vec::new();

    for candidate in candidates {
        let candidate = candidate?;
        let specs = PyIterator::from_bound_object(&candidate)?;
        let mut path_novelty_total = 0.0;
        let mut data_weighted_total = 0.0;
        let mut finding_yield_total = 0.0;
        let mut feature_saturation_total = 0.0;
        let mut profile_saturation_penalty = 0.0;
        let mut path_novelty_count: usize = 0;
        let mut data_novelty_count: usize = 0;
        let mut online_weight_total = 0.0;
        let mut online_weight_max = 1.0;

        for spec in specs {
            let (
                feature,
                finding_weight_base,
                saturation_weight_base,
                path_weight_base,
                data_weight_base,
                maybe_multiplier,
                is_mixed_profile,
            ): (String, f64, f64, f64, f64, Option<f64>, bool) = spec?.extract()?;
            let count = *feature_counts.get(&feature).unwrap_or(&0.0);
            let multiplier_value = maybe_multiplier.unwrap_or(1.0);
            if maybe_multiplier.is_some() {
                online_weight_total += multiplier_value;
                if multiplier_value > online_weight_max {
                    online_weight_max = multiplier_value;
                }
            }
            if path_weight_base > 0.0 {
                path_novelty_total += (path_weight_base * multiplier_value) / (1.0 + count);
                if count == 0.0 {
                    path_novelty_count += 1;
                }
            }
            if data_weight_base > 0.0 {
                data_weighted_total +=
                    (data_weight_base * multiplier_value) * (1.0 + 1.0 / (1.0 + count));
                if count == 0.0 {
                    data_novelty_count += 1;
                }
            }
            let finding_hits = *finding_feature_counts.get(&feature).unwrap_or(&0.0);
            finding_yield_total +=
                bounded_finding_signal(finding_hits) * finding_weight_base * multiplier_value;
            feature_saturation_total +=
                feature_saturation(finding_hits) * saturation_weight_base * multiplier_value;
            if is_mixed_profile {
                profile_saturation_penalty += profile_saturation(count);
            }
        }

        out.push((
            path_novelty_total,
            data_weighted_total,
            finding_yield_total,
            feature_saturation_total,
            profile_saturation_penalty,
            path_novelty_count,
            data_novelty_count,
            online_weight_total,
            online_weight_max,
        ));
    }

    Ok(out)
}

#[pyfunction]
fn sorted_canonical_rows(py: Python<'_>, rows: PyObject) -> PyResult<Vec<String>> {
    let mut stable = stable_rows(py, rows)?;
    stable.sort();
    Ok(stable)
}

#[pyfunction]
fn dedupe_canonical_rows(py: Python<'_>, rows: PyObject) -> PyResult<Vec<String>> {
    let stable = stable_rows(py, rows)?;
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let mut unique: Vec<String> = Vec::new();
    for row in stable {
        let entry = seen.entry(row.clone()).or_insert(0usize);
        if *entry == 0 {
            unique.push(row);
        }
        *entry += 1;
    }
    Ok(unique)
}

#[pyfunction]
fn ordered_row_signature(py: Python<'_>, rows: PyObject) -> PyResult<String> {
    let stable = stable_rows(py, rows)?;
    ordered_key_signature(&stable)
}

#[pyfunction]
fn ordered_row_signatures(py: Python<'_>, row_sets: PyObject) -> PyResult<Vec<String>> {
    Ok(
        parse_row_set_profiles(py_to_json(row_sets.bind(py))?, "row_sets must be a JSON array")?
            .into_iter()
            .map(|profile| profile.ordered_signature)
            .collect()
    )
}

#[pyfunction]
fn has_duplicate_rows(py: Python<'_>, rows: PyObject) -> PyResult<bool> {
    let stable = stable_rows(py, rows)?;
    let mut counts = BTreeMap::new();
    for row in stable {
        let entry = counts.entry(row).or_insert(0usize);
        *entry += 1;
        if *entry > 1 {
            return Ok(true);
        }
    }
    Ok(false)
}

#[pyfunction]
fn unordered_row_signature(py: Python<'_>, rows: PyObject) -> PyResult<String> {
    let mut stable = stable_rows(py, rows)?;
    stable.sort();
    ordered_key_signature(&stable)
}

#[pyfunction]
fn unordered_row_signatures(py: Python<'_>, row_sets: PyObject) -> PyResult<Vec<String>> {
    Ok(
        parse_row_set_profiles(py_to_json(row_sets.bind(py))?, "row_sets must be a JSON array")?
            .into_iter()
            .map(|profile| profile.unordered_signature)
            .collect()
    )
}

#[pyfunction]
fn row_profile(py: Python<'_>, rows: PyObject) -> PyResult<(String, String, bool)> {
    row_profile_from_keys(&stable_rows(py, rows)?)
}

#[pyfunction]
fn row_profiles(py: Python<'_>, row_sets: PyObject) -> PyResult<Vec<(String, String, bool)>> {
    Ok(
        parse_row_set_profiles(py_to_json(row_sets.bind(py))?, "row_sets must be a JSON array")?
            .into_iter()
            .map(|profile| (profile.ordered_signature, profile.unordered_signature, profile.has_duplicates))
            .collect()
    )
}

#[pyfunction]
fn canonicalize_row_order(py: Python<'_>, rows: PyObject) -> PyResult<(Vec<usize>, Vec<String>, String, bool)> {
    let stable = stable_rows(py, rows)?;
    let (indices, ordered_keys) = sorted_key_order(&stable);
    let ordered_signature = ordered_key_signature(&ordered_keys)?;
    let has_duplicates = has_duplicate_key_runs(&ordered_keys);
    Ok((indices, ordered_keys, ordered_signature, has_duplicates))
}

#[pyfunction]
fn row_set_profiles(py: Python<'_>, row_sets: PyObject) -> PyResult<Vec<(String, String)>> {
    Ok(
        parse_row_set_profiles(py_to_json(row_sets.bind(py))?, "row_sets must be a JSON array")?
            .into_iter()
            .map(|profile| (profile.ordered_signature, profile.unordered_signature))
            .collect()
    )
}

#[pyfunction]
fn multiset_row_diff(py: Python<'_>, left_rows: PyObject, right_rows: PyObject) -> PyResult<(Vec<String>, Vec<String>)> {
    let left = stable_rows(py, left_rows)?;
    let right = stable_rows(py, right_rows)?;
    Ok(multiset_diff_from_keys(&left, &right))
}

#[pyfunction]
fn compare_row_sets_summary(
    py: Python<'_>,
    left_rows: PyObject,
    right_rows: PyObject,
) -> PyResult<(String, bool, bool, bool, Vec<String>, Vec<String>)> {
    let left = stable_rows(py, left_rows)?;
    let right = stable_rows(py, right_rows)?;
    let same_row_count = left.len() == right.len();
    if !same_row_count {
        return Ok((
            "row_count".to_string(),
            same_row_count,
            false,
            false,
            Vec::new(),
            Vec::new(),
        ));
    }

    let left_profile = row_profile_from_keys(&left)?;
    let right_profile = row_profile_from_keys(&right)?;
    let same_ordered_rows = left_profile.0 == right_profile.0;
    if same_ordered_rows {
        return Ok(("none".to_string(), true, true, true, Vec::new(), Vec::new()));
    }
    let same_unordered_rows = left_profile.1 == right_profile.1;
    if same_unordered_rows {
        return Ok(("row_order".to_string(), true, false, true, Vec::new(), Vec::new()));
    }
    let (left_only, right_only) = multiset_diff_from_keys(&left, &right);
    Ok(("value".to_string(), true, false, false, left_only, right_only))
}

#[pyfunction]
fn compare_row_sets(py: Python<'_>, left_rows: PyObject, right_rows: PyObject) -> PyResult<(String, bool, bool, Vec<String>, Vec<String>)> {
    let (mismatch_class, same_row_count, same_ordered_rows, _, left_only, right_only) =
        compare_row_sets_summary(py, left_rows, right_rows)?;
    Ok((
        mismatch_class,
        same_row_count,
        same_ordered_rows,
        left_only,
        right_only,
    ))
}

#[pyfunction]
fn compare_row_set_batch(py: Python<'_>, row_sets: PyObject) -> PyResult<(Vec<usize>, String)> {
    let profiles = parse_row_set_profiles(py_to_json(row_sets.bind(py))?, "row_sets must be a JSON array")?;
    if profiles.is_empty() {
        return Ok((Vec::new(), "none".to_string()));
    }
    Ok((row_group_ids_from_profiles(&profiles), row_mismatch_class_from_profiles(&profiles)))
}

#[pyfunction]
fn compare_row_set_batch_summary(py: Python<'_>, row_sets: PyObject) -> PyResult<(Vec<usize>, String, Vec<usize>, bool, Vec<usize>)> {
    let profiles = parse_row_set_profiles(py_to_json(row_sets.bind(py))?, "row_sets must be a JSON array")?;
    if profiles.is_empty() {
        return Ok((Vec::new(), "none".to_string(), Vec::new(), false, Vec::new()));
    }
    let group_ids = row_group_ids_from_profiles(&profiles);
    let majority_group = majority_group_from_group_ids(&group_ids).unwrap_or_default();
    let has_clear_majority = !majority_group.is_empty();
    Ok((
        group_ids.clone(),
        row_mismatch_class_from_profiles(&profiles),
        suspicious_indices_from_group_ids(&group_ids),
        has_clear_majority,
        majority_group,
    ))
}

#[pyfunction]
fn compare_result_batch(py: Python<'_>, results: PyObject) -> PyResult<(Vec<usize>, String)> {
    let profiles = parse_result_profiles(py_to_json(results.bind(py))?)?;
    if profiles.is_empty() {
        return Ok((Vec::new(), "none".to_string()));
    }
    let comparison_keys = profiles
        .iter()
        .map(|profile| profile.comparison_key.clone())
        .collect::<Vec<String>>();
    let group_ids = ordered_group_ids_from_signatures(&comparison_keys);
    let first_status = &profiles[0].status;
    if profiles.iter().any(|profile| &profile.status != first_status) {
        return Ok((group_ids, "status".to_string()));
    }
    if first_status != "ok" {
        let first_error_type = &profiles[0].error_type;
        if profiles.iter().any(|profile| &profile.error_type != first_error_type) {
            return Ok((group_ids, "error_type".to_string()));
        }
        if comparison_keys.iter().all(|key| key == &comparison_keys[0]) {
            return Ok((group_ids, "none".to_string()));
        }
        return Ok((group_ids, "status".to_string()));
    }

    let first_columns = &profiles[0].columns;
    if profiles.iter().any(|profile| &profile.columns != first_columns) {
        return Ok((group_ids, "schema".to_string()));
    }

    let row_profiles = profiles
        .iter()
        .map(|profile| RowSetProfileRecord {
            ordered_signature: profile.row_profile.ordered_signature.clone(),
            unordered_signature: profile.row_profile.unordered_signature.clone(),
            has_duplicates: profile.row_profile.has_duplicates,
            row_count: profile.row_profile.row_count,
        })
        .collect::<Vec<RowSetProfileRecord>>();
    Ok((group_ids, row_mismatch_class_from_profiles(&row_profiles)))
}

#[pyfunction]
fn compare_result_batch_summary(py: Python<'_>, results: PyObject) -> PyResult<(Vec<usize>, String, Vec<usize>, bool, Vec<usize>)> {
    let (group_ids, mismatch_class) = compare_result_batch(py, results)?;
    let majority_group = majority_group_from_group_ids(&group_ids).unwrap_or_default();
    let has_clear_majority = !majority_group.is_empty();
    Ok((
        group_ids.clone(),
        mismatch_class,
        suspicious_indices_from_group_ids(&group_ids),
        has_clear_majority,
        majority_group,
    ))
}

#[pyfunction]
fn compare_result_batch_anchor_summary(
    py: Python<'_>,
    results: PyObject,
    anchor_index: usize,
) -> PyResult<(Vec<usize>, Vec<usize>, String)> {
    let (group_ids, _) = compare_result_batch(py, results)?;
    let matching_indices = anchor_matching_indices_from_group_ids(&group_ids, anchor_index)?;
    let mismatching_indices = anchor_mismatching_indices_from_group_ids(&group_ids, anchor_index)?;
    let confidence = anchor_confidence_from_group_ids(&group_ids, anchor_index)?;
    Ok((matching_indices, mismatching_indices, confidence))
}

#[pyfunction]
fn profile_result_batch(py: Python<'_>, results: PyObject) -> PyResult<Vec<(String, Vec<String>, String, String, String, bool, String)>> {
    Ok(parse_result_profiles(py_to_json(results.bind(py))?)?
        .into_iter()
        .map(|profile| {
            (
                profile.status,
                profile.columns,
                profile.error_type,
                profile.row_profile.ordered_signature,
                profile.row_profile.unordered_signature,
                profile.row_profile.has_duplicates,
                profile.comparison_key,
            )
        })
        .collect())
}

#[pymodule]
fn rust_kernel(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(json_canonical_dumps, m)?)?;
    m.add_function(wrap_pyfunction!(short_sha256_hex, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_sort_key, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_group_keys, m)?)?;
    m.add_function(wrap_pyfunction!(stable_rows, m)?)?;
    m.add_function(wrap_pyfunction!(extract_case_features, m)?)?;
    m.add_function(wrap_pyfunction!(compute_minhash, m)?)?;
    m.add_function(wrap_pyfunction!(score_candidate_feature_metrics_batch, m)?)?;
    m.add_function(wrap_pyfunction!(sorted_canonical_rows, m)?)?;
    m.add_function(wrap_pyfunction!(dedupe_canonical_rows, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_row_signature, m)?)?;
    m.add_function(wrap_pyfunction!(ordered_row_signatures, m)?)?;
    m.add_function(wrap_pyfunction!(has_duplicate_rows, m)?)?;
    m.add_function(wrap_pyfunction!(unordered_row_signature, m)?)?;
    m.add_function(wrap_pyfunction!(unordered_row_signatures, m)?)?;
    m.add_function(wrap_pyfunction!(row_profile, m)?)?;
    m.add_function(wrap_pyfunction!(row_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_row_order, m)?)?;
    m.add_function(wrap_pyfunction!(row_set_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(multiset_row_diff, m)?)?;
    m.add_function(wrap_pyfunction!(compare_row_sets_summary, m)?)?;
    m.add_function(wrap_pyfunction!(compare_row_sets, m)?)?;
    m.add_function(wrap_pyfunction!(compare_row_set_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compare_row_set_batch_summary, m)?)?;
    m.add_function(wrap_pyfunction!(compare_result_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compare_result_batch_summary, m)?)?;
    m.add_function(wrap_pyfunction!(compare_result_batch_anchor_summary, m)?)?;
    m.add_function(wrap_pyfunction!(profile_result_batch, m)?)?;
    Ok(())
}
