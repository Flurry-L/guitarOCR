use guitarocr_backend::project::{import_project, ImportOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::Path;
use tempfile::TempDir;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

const SID: &str = "0123456789abcdef0123456789abcdef";
fn options() -> ImportOptions {
    ImportOptions {
        session_id: Some(SID.into()),
        ..Default::default()
    }
}
fn state() -> Value {
    json!({"id":"old", "pages":[{"image":"project://pages/1.png"}], "revision":19})
}
fn package(state: Value, files: &[(&str, &[u8])]) -> Value {
    json!({"format":"guitarocr-project", "version":1, "session":state,
        "files":files.iter().map(|(name, data)| json!({"path":name,"sha256":format!("{:x}",Sha256::digest(data))})).collect::<Vec<_>>()})
}
fn archive(manifest: Value, files: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
    let opt = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    for (name, data) in files {
        writer.start_file(*name, opt).unwrap();
        writer.write_all(data).unwrap();
    }
    writer.start_file("project.json", opt).unwrap();
    serde_json::to_writer(&mut writer, &manifest).unwrap();
    writer.finish().unwrap().into_inner()
}
fn fixture() -> (TempDir, std::path::PathBuf, std::path::PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("project.zip");
    let workspace = temp.path().join("workspace");
    fs::create_dir(&workspace).unwrap();
    (temp, source, workspace)
}
fn rejects(data: &[u8]) -> String {
    let (_temp, source, workspace) = fixture();
    fs::write(&source, data).unwrap();
    let error = import_project(&source, &workspace, &options())
        .unwrap_err()
        .to_string();
    assert_eq!(
        fs::read_dir(&workspace).unwrap().count(),
        0,
        "left a partial project: {error}"
    );
    error
}
fn portable(value: &mut Value, root: &Path) {
    match value {
        Value::Object(object) => {
            for value in object.values_mut() {
                portable(value, root);
            }
        }
        Value::Array(array) => {
            for value in array {
                portable(value, root);
            }
        }
        Value::String(text) => {
            if let Ok(relative) = Path::new(text).strip_prefix(root) {
                *text = format!(
                    "project://{}",
                    relative.to_str().unwrap().replace('\\', "/")
                );
            }
        }
        _ => {}
    }
}

#[test]
fn imports_the_real_python_export_and_roundtrips_every_json_value() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ui/workbench/examples/Harbor-Light-synthetic-project.zip");
    let temp = tempfile::tempdir().unwrap();
    let imported = import_project(&source, temp.path(), &options()).unwrap();
    assert_eq!(imported.session_id, SID);
    assert_eq!(imported.session["revision"], 0);
    assert_eq!(imported.session["pages"].as_array().unwrap().len(), 1);
    let saved: Value =
        serde_json::from_slice(&fs::read(imported.directory.join("session.json")).unwrap())
            .unwrap();
    assert_eq!(saved, imported.session);
    let recognition: Value =
        serde_json::from_slice(&fs::read(saved["recognition"].as_str().unwrap()).unwrap()).unwrap();
    assert_eq!(recognition["records"].as_array().unwrap().len(), 16);
    assert_eq!(recognition["parts"].as_array().unwrap().len(), 2);
    assert_eq!(recognition["provenance"]["ocr_executed"], false);
    let mut original = ZipArchive::new(fs::File::open(source).unwrap()).unwrap();
    let mut count = 0;
    for i in 0..original.len() {
        let mut entry = original.by_index(i).unwrap();
        let mut before = Vec::new();
        entry.read_to_end(&mut before).unwrap();
        if entry.name() == "project.json" {
            let mut expected = serde_json::from_slice::<Value>(&before).unwrap()["session"].clone();
            expected["id"] = json!(SID);
            expected["revision"] = json!(0);
            expected.as_object_mut().unwrap().remove("ocr_task");
            let mut actual = saved.clone();
            portable(&mut actual, &imported.directory);
            assert_eq!(actual, expected);
            continue;
        }
        count += 1;
        let after = fs::read(imported.directory.join(entry.name())).unwrap();
        if entry.name().ends_with(".json") {
            let mut actual: Value = serde_json::from_slice(&after).unwrap();
            portable(&mut actual, &imported.directory);
            assert_eq!(
                actual,
                serde_json::from_slice::<Value>(&before).unwrap(),
                "{}",
                entry.name()
            );
        } else {
            assert_eq!(after, before, "{}", entry.name());
        }
    }
    assert_eq!(imported.imported_files, count);
}

#[test]
fn preserves_unknown_fields_large_numbers_and_jsonl_remaps() {
    let (_temp, source, workspace) = fixture();
    let recognition = br#"{"schema_version":"1.0","stage":"measure_ocr","records":[{"page":1,"image":"project://pages/1.png","target":"M2 time=4/4 | V0{@0:q:s1f0 @960:e[3:2]:^}","future":{"number":123456789012345678901234567890,"float":0.1234567890123456789,"list":[true,null]}}],"future_schema":{"version":12}}"#;
    let log = b"\r\n{\"image\":\"project://pages/1.png\",\"extra\":[1,2,3]}\r\n\n";
    let files: Vec<(&str, &[u8])> = vec![
        ("pages/1.png", b"image"),
        ("recognition/manifest.json", recognition),
        ("recognition/history.JSONL", log),
    ];
    let mut state = state();
    state["recognition"] = json!("project://recognition/manifest.json");
    state["ocr_task"] = json!({"checkpoint":"originating-machine/signature"});
    state["future_session"] = json!({"favorite":"blue", "extra_path":"project://pages/1.png"});
    fs::write(&source, archive(package(state.clone(), &files), &files)).unwrap();
    let imported = import_project(&source, &workspace, &options()).unwrap();
    assert!(imported.session.get("ocr_task").is_none());
    let mut restored = imported.session.clone();
    portable(&mut restored, &imported.directory);
    assert_eq!(restored["future_session"], state["future_session"]);
    let mut restored: Value = serde_json::from_slice(
        &fs::read(imported.directory.join("recognition/manifest.json")).unwrap(),
    )
    .unwrap();
    portable(&mut restored, &imported.directory);
    assert_eq!(
        restored,
        serde_json::from_slice::<Value>(recognition).unwrap()
    );
    let history = fs::read_to_string(imported.directory.join("recognition/history.JSONL")).unwrap();
    assert_eq!(history.lines().count(), 1);
    let mut row: Value = serde_json::from_str(&history).unwrap();
    portable(&mut row, &imported.directory);
    assert_eq!(
        row,
        json!({"image":"project://pages/1.png","extra":[1,2,3]})
    );
}

#[test]
fn generates_distinct_valid_ids_and_never_overwrites_existing_project() {
    let (_temp, source, workspace) = fixture();
    let files: Vec<(&str, &[u8])> = vec![("pages/1.png", b"image")];
    fs::write(&source, archive(package(state(), &files), &files)).unwrap();
    let first = import_project(&source, &workspace, &Default::default()).unwrap();
    let second = import_project(&source, &workspace, &Default::default()).unwrap();
    assert_ne!(first.session_id, second.session_id);
    assert_eq!(first.session_id.len(), 32);
    let saved = fs::read(first.directory.join("session.json")).unwrap();
    let opts = ImportOptions {
        session_id: Some(first.session_id),
        ..Default::default()
    };
    assert!(import_project(&source, &workspace, &opts).is_err());
    assert_eq!(
        fs::read(first.directory.join("session.json")).unwrap(),
        saved
    );
    for id in [
        "../escape",
        "A123456789abcdef0123456789abcdef",
        "",
        "0123456789abcdef0123456789abcdef/",
    ] {
        assert!(import_project(
            &source,
            &workspace,
            &ImportOptions {
                session_id: Some(id.into()),
                ..Default::default()
            }
        )
        .is_err());
    }
}

#[test]
fn rejects_slip_alias_reserved_device_and_directory_entries() {
    for name in [
        "../escape",
        "/absolute",
        "C:/absolute",
        "a\\b",
        "./a",
        "a//b",
        "a/../b",
        "a/",
        "Session.json",
        "JOB.JSON",
        "nul.txt",
        "COM1",
        "trailing.",
        "space ",
        "bad\0name",
        "stream:target",
    ] {
        let files: Vec<(&str, &[u8])> = vec![(name, b"bad")];
        assert!(
            !rejects(&archive(package(state(), &files), &files)).is_empty(),
            "{name}"
        );
    }
    for names in [
        ["a.txt", "A.txt"],
        ["stra\u{df}e.txt", "STRASSE.txt"],
        ["A/x", "a/y"],
        ["a", "a/b"],
    ] {
        let files: Vec<(&str, &[u8])> = names.iter().map(|name| (*name, &b"bad"[..])).collect();
        rejects(&archive(package(state(), &files), &files));
    }
}

#[test]
fn rejects_exact_duplicate_members_even_when_zip_lookup_collapses_them() {
    let files: Vec<(&str, &[u8])> = vec![("sameA", b"a"), ("sameB", b"b")];
    let mut data = archive(package(state(), &files), &files);
    for i in 0..data.len() - 5 {
        if &data[i..i + 5] == b"sameB" {
            data[i + 4] = b'A';
        }
    }
    assert!(rejects(&data).contains("Duplicate"));
}

#[test]
fn rejects_symlinks_and_special_file_modes() {
    for mode in [0o120777u32, 0o010666u32, 0o040777u32] {
        let files: Vec<(&str, &[u8])> = vec![("pages/1.png", b"target")];
        let mut data = archive(package(state(), &files), &files);
        let i = data.windows(4).position(|w| w == b"PK\x01\x02").unwrap();
        data[i + 5] = 3; // UNIX creator; the mode is in the upper external-attribute bits.
        data[i + 38..i + 42].copy_from_slice(&(mode << 16).to_le_bytes());
        rejects(&data);
    }
}

#[test]
fn rejects_corruption_hash_mismatch_manifest_mismatch_and_missing_files() {
    let files: Vec<(&str, &[u8])> = vec![("pages/1.png", b"image")];
    let manifest = package(state(), &files);
    rejects(b"not a ZIP");
    let mut data = archive(manifest.clone(), &files);
    data.truncate(data.len() - 2);
    rejects(&data);
    for malformed in [
        Value::Null,
        json!([]),
        json!({"format":"guitarocr-project","version":2}),
        json!({"format":"guitarocr-project","version":1,"files":[],"session":[]}),
    ] {
        rejects(&archive(malformed, &files));
    }
    let mut wrong_hash = manifest.clone();
    wrong_hash["files"][0]["sha256"] = json!("a".repeat(64));
    assert!(rejects(&archive(wrong_hash, &files)).contains("SHA-256 mismatch"));
    let mut wrong_files = manifest.clone();
    wrong_files["files"] = json!([]);
    rejects(&archive(wrong_files, &files));
    let mut wrong_page = state();
    wrong_page["pages"][0]["image"] = json!("project://missing.png");
    rejects(&archive(package(wrong_page, &files), &files));
    let mut data = archive(manifest, &files);
    // Corrupt the first stored compressed stream, leaving ZIP structure intact.
    let name_len = u16::from_le_bytes([data[26], data[27]]) as usize;
    let extra_len = u16::from_le_bytes([data[28], data[29]]) as usize;
    data[30 + name_len + extra_len] ^= 0xff;
    rejects(&data);
}

#[test]
fn rejects_external_references_invalid_page_numbers_and_malformed_jsonl() {
    let page: (&str, &[u8]) = ("pages/1.png", b"image");
    for value in [
        json!("/outside.png"),
        json!("https://example.org/image"),
        json!("project://../escape"),
        json!("project://pages/missing.png"),
    ] {
        let mut s = state();
        s["pages"][0]["image"] = value;
        rejects(&archive(package(s, &[page]), &[page]));
    }
    for value in [
        json!(0),
        json!(2),
        json!(-1),
        json!(1.0),
        json!(true),
        json!("1"),
        Value::Null,
    ] {
        let mut s = state();
        s["boxes"] = json!([{"page":value}]);
        rejects(&archive(package(s, &[page]), &[page]));
    }
    for content in [
        &b"{\"page\":2}"[..],
        &b"{\"image\":\"/outside\"}"[..],
        &b"{\"other\":\"project://../escape\"}"[..],
        &b"{}\n{\"partial\":"[..],
        &b"{\"text\":\"\xff\"}"[..],
    ] {
        let files = [page, ("history.jsonl", content)];
        rejects(&archive(package(state(), &files), &files));
    }
}

#[test]
fn validates_stage_schema_and_recognition_targets() {
    let page: (&str, &[u8]) = ("pages/1.png", b"image");
    let mut s = state();
    s["recognition"] = json!("project://recognition.json");
    for target in [
        "no M2",
        "M2 | V0{@0:q:^}",
        "M2 | V0{@0:d0:s1f0}",
        "M2 | V0{@0:e[3:0]:s1f0}",
        "M2 | V0{@0:q:invalid}",
        "M2 unknown=x | V0{}",
    ] {
        let record = serde_json::to_vec(&json!({"schema_version":"1.0","stage":"measure_ocr", "records":[{"target":target,"image":"project://pages/1.png"}]})).unwrap();
        let files = [page, ("recognition.json", record.as_slice())];
        rejects(&archive(package(s.clone(), &files), &files));
    }
    for invalid in [
        json!({"schema_version":"2.0","stage":"measure_ocr","records":[]}),
        json!({"schema_version":"1.0","stage":"layout","records":[]}),
        json!({"schema_version":"1.0","stage":"measure_ocr","records":null}),
    ] {
        let content = serde_json::to_vec(&invalid).unwrap();
        let files = [page, ("recognition.json", content.as_slice())];
        rejects(&archive(package(s.clone(), &files), &files));
    }
    s["recognition"] = json!("project://manifest.txt");
    let files = [page, ("manifest.txt", &b"{}"[..])];
    rejects(&archive(package(s, &files), &files));
}

#[test]
fn enforces_caller_page_and_byte_limits_before_materialization() {
    let (_temp, source, workspace) = fixture();
    let files: Vec<(&str, &[u8])> = vec![("pages/1.png", b"image")];
    fs::write(&source, archive(package(state(), &files), &files)).unwrap();
    for options in [
        ImportOptions {
            max_bytes: 1,
            ..options()
        },
        ImportOptions {
            max_pages: 0,
            ..options()
        },
    ] {
        assert!(import_project(&source, &workspace, &options).is_err());
        assert_eq!(fs::read_dir(&workspace).unwrap().count(), 0);
    }
    // Header bounds apply even to a tiny highly compressed input.
    let mut data = fs::read(&source).unwrap();
    let i = data.windows(4).position(|w| w == b"PK\x01\x02").unwrap();
    data[i + 24..i + 28].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(rejects(&data).contains("byte limit"));
    let mut data = fs::read(&source).unwrap();
    let i = data.windows(4).rposition(|w| w == b"PK\x05\x06").unwrap();
    data[i + 8..i + 10].copy_from_slice(&25001u16.to_le_bytes());
    data[i + 10..i + 12].copy_from_slice(&25001u16.to_le_bytes());
    rejects(&data);
}

#[cfg(unix)]
#[test]
fn does_not_follow_or_delete_existing_destination_symlink() {
    use std::os::unix::fs::symlink;
    let (_temp, source, workspace) = fixture();
    let files: Vec<(&str, &[u8])> = vec![("pages/1.png", b"image")];
    fs::write(&source, archive(package(state(), &files), &files)).unwrap();
    let elsewhere = workspace.parent().unwrap().join("nonexistent");
    symlink(&elsewhere, workspace.join(SID)).unwrap();
    assert!(import_project(&source, &workspace, &options()).is_err());
    assert!(workspace
        .join(SID)
        .symlink_metadata()
        .unwrap()
        .file_type()
        .is_symlink());
    assert!(!elsewhere.exists());
}

#[test]
fn native_import_export_import_preserves_the_entire_real_demo() {
    use guitarocr_backend::project::{export_project, load_session, write_session_atomic};
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ui/workbench/examples/Harbor-Light-synthetic-project.zip");
    let temp = tempfile::tempdir().unwrap();
    let first_workspace = temp.path().join("first");
    let first = import_project(&source, &first_workspace, &options()).unwrap();
    let mut changed = load_session(&first_workspace, SID).unwrap();
    changed["future"] = json!({"nested":["preserve",true,99]});
    changed["ocr_task"] = json!({"checkpoint":"origin machine"});
    changed["revision"] = json!(123);
    write_session_atomic(&first_workspace, SID, &changed).unwrap();
    assert_eq!(load_session(&first_workspace, SID).unwrap(), changed);
    fs::write(first.directory.join("job.json"), b"{}\n").unwrap();
    fs::write(first.directory.join("ignored.tmp"), b"temporary").unwrap();
    fs::write(first.directory.join("ignored.zip"), b"archive").unwrap();
    fs::create_dir(first.directory.join("tmp")).unwrap();
    fs::write(first.directory.join("tmp/secret.json"), b"{}\n").unwrap();
    let destination = temp.path().join("native-export.zip");
    let exported = export_project(&first_workspace, SID, &destination).unwrap();
    let mut zip = ZipArchive::new(fs::File::open(&exported).unwrap()).unwrap();
    for ignored in [
        "session.json",
        "job.json",
        "ignored.tmp",
        "ignored.zip",
        "tmp/secret.json",
    ] {
        assert!(zip.by_name(ignored).is_err());
    }
    let new_id = "fedcba9876543210fedcba9876543210";
    let imported = import_project(
        exported,
        temp.path().join("second"),
        &ImportOptions {
            session_id: Some(new_id.into()),
            ..Default::default()
        },
    )
    .unwrap();
    let mut normalized_first = changed;
    portable(&mut normalized_first, &first.directory);
    normalized_first["id"] = json!(new_id);
    normalized_first["revision"] = json!(0);
    normalized_first.as_object_mut().unwrap().remove("ocr_task");
    let mut normalized_second = imported.session.clone();
    portable(&mut normalized_second, &imported.directory);
    assert_eq!(normalized_second, normalized_first);
    for i in 0..zip.len() {
        let entry = zip.by_index(i).unwrap();
        if entry.name() == "project.json" {
            continue;
        }
        let before = fs::read(first.directory.join(entry.name())).unwrap();
        let after = fs::read(imported.directory.join(entry.name())).unwrap();
        if entry.name().ends_with(".json") {
            let mut before: Value = serde_json::from_slice(&before).unwrap();
            portable(&mut before, &first.directory);
            let mut after: Value = serde_json::from_slice(&after).unwrap();
            portable(&mut after, &imported.directory);
            assert_eq!(before, after);
        } else {
            assert_eq!(before, after);
        }
    }
}

#[test]
fn export_recovers_only_interrupted_last_journal_record_without_mutation() {
    use guitarocr_backend::project::export_project;
    let (_temp, source, workspace) = fixture();
    let files: Vec<(&str, &[u8])> = vec![("pages/1.png", b"image")];
    fs::write(&source, archive(package(state(), &files), &files)).unwrap();
    let imported = import_project(&source, &workspace, &options()).unwrap();
    let output = workspace.parent().unwrap().join("export.zip");
    let log = imported.directory.join("recognition.jsonl");
    for tail in [&b"{\"partial\":"[..], &b"{\"partial\":\"\xe4\xb8"[..]] {
        let mut contents = b"{\"extra\":true}\n".to_vec();
        contents.extend_from_slice(tail);
        fs::write(&log, &contents).unwrap();
        export_project(&workspace, SID, &output).unwrap();
        assert_eq!(fs::read(&log).unwrap(), contents);
        let mut zip = ZipArchive::new(fs::File::open(&output).unwrap()).unwrap();
        let mut recovered = String::new();
        zip.by_name("recognition.jsonl")
            .unwrap()
            .read_to_string(&mut recovered)
            .unwrap();
        assert_eq!(recovered, "{\"extra\":true}\n");
    }
    let old_output = fs::read(&output).unwrap();
    for bad in [&b"{\"partial\":\n"[..], &b"bad\n{}\n"[..]] {
        fs::write(&log, bad).unwrap();
        assert!(export_project(&workspace, SID, &output).is_err());
        assert_eq!(fs::read(&output).unwrap(), old_output);
        assert!(
            !fs::read_dir(workspace.parent().unwrap()).unwrap().any(|e| e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".guitarocr-"))
        );
    }
}

#[test]
fn session_save_rejects_mismatched_id_without_changing_existing_state() {
    use guitarocr_backend::project::{load_session, write_session_atomic};
    let (_temp, source, workspace) = fixture();
    let files: Vec<(&str, &[u8])> = vec![("pages/1.png", b"image")];
    fs::write(&source, archive(package(state(), &files), &files)).unwrap();
    let imported = import_project(&source, &workspace, &options()).unwrap();
    let mut wrong = imported.session.clone();
    wrong["id"] = json!("other");
    assert!(write_session_atomic(&workspace, SID, &wrong).is_err());
    assert_eq!(load_session(&workspace, SID).unwrap(), imported.session);
    assert_eq!(fs::read_dir(imported.directory).unwrap().count(), 2);
}

#[cfg(unix)]
#[test]
fn export_omits_symlinks_and_loading_rejects_symlink_sessions() {
    use guitarocr_backend::project::{export_project, load_session, write_session_atomic};
    use std::os::unix::fs::symlink;
    let (_temp, source, workspace) = fixture();
    let files: Vec<(&str, &[u8])> = vec![("pages/1.png", b"image")];
    fs::write(&source, archive(package(state(), &files), &files)).unwrap();
    let imported = import_project(&source, &workspace, &options()).unwrap();
    symlink(&source, imported.directory.join("link.txt")).unwrap();
    let output = workspace.parent().unwrap().join("export.zip");
    export_project(&workspace, SID, &output).unwrap();
    let mut zip = ZipArchive::new(fs::File::open(&output).unwrap()).unwrap();
    assert!(zip.by_name("link.txt").is_err());
    fs::remove_file(imported.directory.join("session.json")).unwrap();
    symlink(&source, imported.directory.join("session.json")).unwrap();
    assert!(load_session(&workspace, SID).is_err());
    assert!(write_session_atomic(&workspace, SID, &imported.session).is_err());
}

#[test]
fn native_ocr_archives_discard_machine_context_without_weakening_path_checks() {
    use guitarocr_backend::project::export_project;
    let (temp, source, workspace) = fixture();
    let layout = br#"{"schema_version":"1.0","stage":"layout"}"#;
    let info = br#"{"schema_version":"1.0","stage":"document_info"}"#;
    let recognition = br#"{"schema_version":"1.0","stage":"measure_ocr","records":[{"page":1,"image":"project://pages/1.png","target":"M2 time=4/4 | V0{@0:q:s1f0}"}]}"#;
    let context = serde_json::to_vec(&json!({"native_context":1,"layout":"a".repeat(64),"info":"b".repeat(64),"images":["c".repeat(64)]})).unwrap();
    let files: Vec<(&str, &[u8])> = vec![
        ("pages/1.png", b"image"),
        ("layout/manifest.json", layout),
        ("info/manifest.json", info),
        ("ocr/manifest.json", recognition),
        ("ocr/recognition_context.json", &context),
    ];
    let mut state = state();
    state["layout"] = json!("project://layout/manifest.json");
    state["info"] = json!("project://info/manifest.json");
    state["recognition"] = json!("project://ocr/manifest.json");
    let manifest = package(state.clone(), &files);
    fs::write(&source, archive(manifest.clone(), &files)).unwrap();
    let restored = import_project(&source, &workspace, &options()).unwrap();
    assert!(!restored
        .directory
        .join("ocr/recognition_context.json")
        .exists());
    assert_eq!(restored.imported_files, files.len() - 1);
    // A real OCR run's machine-specific context must not enter new portable ZIPs.
    fs::write(
        restored.directory.join("ocr/recognition_context.json"),
        &context,
    )
    .unwrap();
    let exported = export_project(&workspace, SID, temp.path().join("exported.zip")).unwrap();
    let mut zip = ZipArchive::new(fs::File::open(&exported).unwrap()).unwrap();
    assert!(zip.by_name("ocr/recognition_context.json").is_err());
    import_project(&exported, temp.path().join("roundtrip"), &options()).unwrap();
    // Existing archives still verify even discarded checkpoint bytes.
    let mut corrupt = manifest.clone();
    corrupt["files"][4]["sha256"] = json!("0".repeat(64));
    assert!(rejects(&archive(corrupt, &files)).contains("SHA-256 mismatch"));
    // Neither hashes nor discarded files become valid stage/path references.
    state["info"] = json!("b".repeat(64) + ".json");
    assert!(rejects(&archive(package(state.clone(), &files), &files)).contains("inside its ZIP"));
    state["info"] = json!("project://ocr/recognition_context.json");
    assert!(rejects(&archive(package(state, &files), &files)).contains("Missing project reference"));
}
