// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `living::steer`: the vector loader and the site placement.

use super::*;

fn write_v(path: &Path, v: &[f32], meta: &[(&str, &str)]) {
    let mut header = serde_json::Map::new();
    let data: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    header.insert(
        "v".into(),
        serde_json::json!({
            "dtype": "F32", "shape": [v.len()], "data_offsets": [0, data.len()]
        }),
    );
    let meta: serde_json::Map<String, serde_json::Value> = meta
        .iter()
        .map(|(k, v)| (k.to_string(), serde_json::Value::from(*v)))
        .collect();
    header.insert("__metadata__".into(), serde_json::Value::Object(meta));
    let h = serde_json::to_vec(&header).unwrap();
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend(h);
    bytes.extend(data);
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn steer_vectors_are_unit_scaled_by_rms_and_summed() {
    let dir = std::env::temp_dir().join(format!("living-steer-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    write_v(
        &dir.join("a.safetensors"),
        &[3.0, 4.0],
        &[("base", "rev1"), ("layer", "22"), ("rms", "2"), ("topic", "calm")],
    );
    write_v(
        &dir.join("b.safetensors"),
        &[0.0, -10.0],
        &[("base", "rev1"), ("rms", "1"), ("scale", "0.5")],
    );
    std::fs::write(
        dir.join("index.json"),
        r#"{"files": ["a.safetensors", "b.safetensors"]}"#,
    )
    .unwrap();
    let host = read_vectors(&dir, "rev1", 22, 2, None).unwrap();
    // Unit RMS in 2 dims is norm sqrt(2). a: 2 * sqrt2 * (0.6, 0.8);
    // b: 0.5 * 1 * sqrt2 * (0, -1).
    let r2 = 2f32.sqrt();
    assert!(
        (host.v[0] - 1.2 * r2).abs() < 1e-5 && (host.v[1] - 1.1 * r2).abs() < 1e-5,
        "{:?}",
        host.v
    );
    assert_eq!(host.files.len(), 2);
    assert_eq!(host.files[0].topic, "calm");
    assert!(read_vectors(&dir, "rev2", 22, 2, None).is_err());
    assert!(read_vectors(&dir, "rev1", 21, 2, None).is_err());
    std::fs::write(dir.join("index.json"), r#"{"files": ["b.safetensors"]}"#).unwrap();
    write_v(&dir.join("b.safetensors"), &[0.0, -10.0], &[("base", "rev1")]);
    assert!(read_vectors(&dir, "rev1", 22, 2, None).is_err());
    let host = read_vectors(&dir, "rev1", 22, 2, Some(3.0)).unwrap();
    assert!((host.v[1] + 3.0 * 2f32.sqrt()).abs() < 1e-5);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn steer_site_is_the_stream_between_layer_and_next() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.num_hidden_layers = 8;
    config.layer_types = vec![
        LayerType::LinearAttention,
        LayerType::LinearAttention,
        LayerType::LinearAttention,
        LayerType::FullAttention,
        LayerType::LinearAttention,
        LayerType::LinearAttention,
        LayerType::LinearAttention,
        LayerType::FullAttention,
    ];
    assert_eq!(site_for(&config, None).unwrap(), (4, (4, Spot::End)));
    assert_eq!(site_for(&config, Some(2)).unwrap(), (2, (2, Spot::End)));
    assert_eq!(site_for(&config, Some(3)).unwrap(), (3, (4, Spot::Start)));
    assert!(site_for(&config, Some(7)).is_err());
    assert!(site_for(&config, Some(8)).is_err());
}
