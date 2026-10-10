use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=GUITAROCR_BUILD_MODEL_CATALOG");
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .to_owned();
    let catalog = match env::var_os("GUITAROCR_BUILD_MODEL_CATALOG") {
        Some(path) => root.join(path),
        None => root.join("weights/distribution.json"),
    };
    println!("cargo:rerun-if-changed={}", catalog.display());
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("model-distribution.json");
    fs::copy(&catalog, output).unwrap_or_else(|error| {
        panic!("Cannot stage model catalog {}: {error}", catalog.display())
    });
}
