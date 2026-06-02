use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

fn py_to_json(py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<Value> {
    let json = py.import_bound("json")?;
    let dumped: String = json
        .getattr("dumps")?
        .call1((payload,))?
        .extract()?;
    serde_json::from_str(&dumped).map_err(|err| PyValueError::new_err(err.to_string()))
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
    let value = py_to_json(py, payload.bind(py))?;
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
    row_keys_from_value(py_to_json(py, payloads.bind(py))?, "payloads must be a JSON array")
}

#[pyfunction]
fn stable_rows(py: Python<'_>, rows: PyObject) -> PyResult<Vec<String>> {
    row_keys_from_value(py_to_json(py, rows.bind(py))?, "rows must be a JSON array")
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
        parse_row_set_profiles(py_to_json(py, row_sets.bind(py))?, "row_sets must be a JSON array")?
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
        parse_row_set_profiles(py_to_json(py, row_sets.bind(py))?, "row_sets must be a JSON array")?
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
        parse_row_set_profiles(py_to_json(py, row_sets.bind(py))?, "row_sets must be a JSON array")?
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
        parse_row_set_profiles(py_to_json(py, row_sets.bind(py))?, "row_sets must be a JSON array")?
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
    let profiles = parse_row_set_profiles(py_to_json(py, row_sets.bind(py))?, "row_sets must be a JSON array")?;
    if profiles.is_empty() {
        return Ok((Vec::new(), "none".to_string()));
    }
    Ok((row_group_ids_from_profiles(&profiles), row_mismatch_class_from_profiles(&profiles)))
}

#[pyfunction]
fn compare_row_set_batch_summary(py: Python<'_>, row_sets: PyObject) -> PyResult<(Vec<usize>, String, Vec<usize>, bool, Vec<usize>)> {
    let profiles = parse_row_set_profiles(py_to_json(py, row_sets.bind(py))?, "row_sets must be a JSON array")?;
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
    let profiles = parse_result_profiles(py_to_json(py, results.bind(py))?)?;
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
    Ok(parse_result_profiles(py_to_json(py, results.bind(py))?)?
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
