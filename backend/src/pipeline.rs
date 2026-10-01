//! Native application orchestration. Stage results are immutable; a session
//! switches pointers only after success. Journals never replace accepted music.
use crate::assets::NativeAssets;
use crate::auxiliary_onnx::{initialize_runtime, SignatureReader};
use crate::edit;
use crate::layout;
use crate::models::Models;
use crate::project::{load_session, write_session_atomic};
use crate::recognition::{self, ImagePolicy, InformationConfig, MeasureConfig};
use crate::tasks::Progress;
use crate::view;
use guitarocr_engine::runtime::{Options, Runtime, Session};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
type Result<T> = std::result::Result<T, String>;
pub struct Pipeline {
    pub native: NativeAssets,
    pub models: Arc<Models>,
    engine: Mutex<Option<Arc<Runtime>>>,
    acceleration: Option<crate::acceleration::Acceleration>,
    slots: usize,
}
impl Pipeline {
    pub fn new(native: NativeAssets, models: Arc<Models>, slots: usize) -> Self {
        let acceleration = if native.capabilities.cuda == Some(true) || native.llama.is_none() {
            None
        } else {
            crate::acceleration::Acceleration::for_device(&models, slots)
        };
        Self {
            native,
            models,
            engine: Mutex::new(None),
            acceleration,
            slots,
        }
    }
    pub fn available(&self) -> bool {
        self.native.ort.is_some() && self.native.llama.is_some()
    }
    pub fn ready(&self) -> bool {
        self.models.ready()
            && (self.acceleration.as_ref().is_none_or(|a| a.cached())
                || self.engine.try_lock().is_ok_and(|e| e.is_some()))
    }
    pub fn download_bytes(&self) -> u64 {
        self.models.download_bytes()
            + self
                .acceleration
                .as_ref()
                .filter(|a| !a.cached())
                .map_or(0, |a| a.bytes())
    }
    pub fn device(&self) -> Value {
        self.engine
            .try_lock()
            .map(|engine| {
                engine
                    .as_ref()
                    .map_or(json!({"device":"auto"}), |engine| engine.device())
            })
            .unwrap_or(json!({"device":"auto","loading":true}))
    }
    pub fn stats(&self) -> Value {
        self.engine
            .try_lock()
            .ok()
            .and_then(|engine| engine.as_ref().map(|e| e.stats()))
            .unwrap_or(Value::Null)
    }
    pub fn unload(&self) -> Result<()> {
        let engine = self
            .engine
            .try_lock()
            .map_err(|_| "模型正在准备，请先停止任务")?;
        if let Some(runtime) = engine.as_ref() {
            runtime.unload()?;
        }
        Ok(())
    }
    pub fn shutdown(&self) {
        if let Ok(engine) = self.engine.lock() {
            if let Some(engine) = engine.as_ref() {
                engine.shutdown();
            }
        }
    }
    fn with_engine<T>(
        &self,
        allow_download: bool,
        progress: &Progress,
        run: impl FnOnce(&mut Session<'_>) -> Result<T>,
    ) -> Result<T> {
        self.models.ensure(allow_download, progress)?;
        let engine = {
            let mut engine = loop {
                progress.checkpoint()?;
                match self.engine.try_lock() {
                    Ok(engine) => break engine,
                    Err(std::sync::TryLockError::WouldBlock) => {
                        std::thread::sleep(std::time::Duration::from_millis(100))
                    }
                    Err(_) => return Err("识别引擎不可用，请重新打开应用".into()),
                }
            };
            if engine.is_none() {
                let accelerator = match &self.acceleration {
                    Some(acceleration) => match acceleration.prepare(allow_download, progress) {
                        Ok(path) => path,
                        Err(error) => {
                            progress.checkpoint()?;
                            eprintln!("显卡组件准备失败：{error}");
                            progress.update(json!({"message":"显卡加速暂不可用，改用 CPU 识别；重新打开应用可重试下载"}))?;
                            None
                        }
                    },
                    None => None,
                };
                *engine = Some(Arc::new(Runtime::new(Options {
                    executable: self
                        .native
                        .llama
                        .clone()
                        .ok_or("识别组件未准备，请重新安装应用")?,
                    accelerator,
                    model: self.models.model(),
                    projector: self.models.projector(),
                    log: self.models.root.join("inference.log"),
                    capabilities: self.native.capabilities.clone(),
                    slots: self.slots,
                    context: 8192,
                })));
            }
            engine.as_ref().unwrap().clone()
        };
        engine.run(
            &|| progress.cancelled(),
            &|message| progress.update(json!({"message":message,"done":0,"total":0})),
            run,
        )
    }
    pub fn detect(
        &self,
        root: &Path,
        sid: &str,
        requested: &str,
        allow_download: bool,
        progress: &Progress,
    ) -> Result<()> {
        let ort = self.native.ort.as_ref().ok_or("原生ONNX组件未准备")?;
        self.models.ensure(allow_download, progress)?;
        layout::detect(root, sid, ort, &self.models.layout(), requested, progress)
    }
    pub fn information(
        &self,
        root: &Path,
        sid: &str,
        allow_download: bool,
        progress: &Progress,
    ) -> Result<()> {
        let mut state = load_session(root, sid).map_err(|e| e.to_string())?;
        let directory = root.join(sid);
        let (layout_path, layout) = stage(&directory, &state, "layout", "layout")?;
        let output = tempfile::Builder::new()
            .prefix("info_")
            .tempdir_in(&directory)
            .map_err(|e| e.to_string())?;
        let flags = &self.models.capabilities;
        let config = InformationConfig {
            layout_path: layout_path.clone(),
            fallback_title: state["input_names"][0]
                .as_str()
                .unwrap_or("未命名乐谱")
                .into(),
            image_policy: Some(ImagePolicy::from_value(&self.models.policy)?),
            staff_profile: flags["staff_profile"] == true,
            pitch_context: flags["pitch_context"] == true,
            score_structure: flags["score_structure"] == true,
            compact_structure: flags["compact_structure"] == true,
            structure_output: Some(output.path().to_owned()),
            ..Default::default()
        };
        progress.update(json!({"message":"正在读取乐器与谱面信息","done":0,"total":0}))?;
        let mut info = self.with_engine(allow_download, progress, |engine| {
            recognition::recognize_information(engine, &layout, &config, &|| progress.cancelled())
        })?;
        let predictions = output.path().join("predictions.json");
        edit::json_file(&predictions, &json!(info.predictions))?;
        let structure = output.path().join("structure_predictions.json");
        edit::json_file(&structure, &json!(info.structure_predictions))?;
        info.manifest["predictions"] = json!(predictions);
        info.manifest["structure_predictions"] = json!(structure);
        info.manifest["layout"] = json!(layout_path);
        let manifest = output.path().join("info.json");
        edit::json_file(&manifest, &info.manifest)?;
        progress.checkpoint()?;
        let accepted = if let Some(previous) = state["recognition"].as_str() {
            let source = view::read_json(Path::new(previous))?;
            if let Some(mut updated) = recognition::update_information(&source, &info.manifest)? {
                updated["info"] = json!(manifest);
                Some(edit::save_recognition(output.path(), &updated)?)
            } else {
                None
            }
        } else {
            None
        };
        state["info"] = json!(manifest);
        state["recognition"] = json!(accepted);
        state["export"] = Value::Null;
        state.as_object_mut().unwrap().remove("ocr_task");
        bump(&mut state)?;
        progress.checkpoint()?;
        write_session_atomic(root, sid, &state).map_err(|e| e.to_string())?;
        let _ = output.keep();
        Ok(())
    }
    pub fn recognize(
        &self,
        root: &Path,
        sid: &str,
        body: &Value,
        progress: &Progress,
    ) -> Result<()> {
        let allow_download = body["allow_download"] == true;
        let resume = body["resume"] == true;
        let mut state = load_session(root, sid).map_err(|e| e.to_string())?;
        let directory = root.join(sid).canonicalize().map_err(|e| e.to_string())?;
        let (layout_path, layout) = stage(&directory, &state, "layout", "layout")?;
        let (info_path, mut info) = stage(&directory, &state, "info", "document_info")?;
        if info["layout"].as_str() != layout_path.to_str() {
            return Err("谱面信息不属于当前区域版本，请重新读取".into());
        }
        let mut records = recognition::prepare_records(&layout, &info)?;
        if records.is_empty() {
            return Err("请至少添加一个小节框".into());
        }
        let retry: Option<HashSet<usize>> = body
            .get("measures")
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_array()
                    .ok_or("无效的重试小节列表".into())
                    .and_then(|rows| {
                        rows.iter()
                            .map(|n| {
                                n.as_u64()
                                    .filter(|n| *n >= 1 && *n <= records.len() as u64)
                                    .map(|n| n as usize)
                                    .ok_or("无效的重试小节".into())
                            })
                            .collect::<Result<HashSet<_>>>()
                    })
            })
            .transpose()?;
        if retry.as_ref().is_some_and(HashSet::is_empty) {
            return Err("请选择需要重试的小节".into());
        }
        let output = if resume {
            let path = PathBuf::from(
                state["ocr_task"]["output"]
                    .as_str()
                    .ok_or("没有可以继续的任务")?,
            );
            let path = path.canonicalize().map_err(|e| e.to_string())?;
            if !path.starts_with(&directory) {
                return Err("识别任务目录超出项目".into());
            }
            path
        } else {
            let path = tempfile::Builder::new()
                .prefix("ocr_")
                .tempdir_in(&directory)
                .map_err(|e| e.to_string())?
                .keep();
            state["ocr_task"] = json!({"output":path,"measures":body.get("measures").cloned().unwrap_or(Value::Null),"source":if retry.is_some(){state["recognition"].clone()}else{Value::Null}});
            if retry.is_some() && !state["recognition"].is_string() {
                return Err("请先完成一次识别再重试部分小节".into());
            }
            write_session_atomic(root, sid, &state).map_err(|e| e.to_string())?;
            path
        };
        let retry = if resume {
            state["ocr_task"]["measures"].as_array().map(|rows| {
                rows.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect::<HashSet<_>>()
            })
        } else {
            retry
        };
        let initial = if let Some(path) = state["ocr_task"]["source"].as_str() {
            let path = Path::new(path).canonicalize().map_err(|e| e.to_string())?;
            if !path.starts_with(&directory) {
                return Err("识别种子超出项目".into());
            }
            view::read_json(&path)?
        } else {
            Value::Null
        };
        let config = MeasureConfig {
            image_policy: Some(ImagePolicy::from_value(&self.models.policy)?),
            visual_pitch: self.models.capabilities["visual_pitch"] == true,
            pitch_context: self.models.capabilities["pitch_context"] == true,
            written_pitch: self.models.capabilities["written_pitch"] == true,
            constrained_decoding: self.models.capabilities["m2_constraints"] == true,
            retry_constraints: self.models.capabilities["m2_retry_constraints"] == true,
            ..Default::default()
        };
        self.with_engine(allow_download, progress, |_| Ok(()))?;
        let selected_device = self.device();
        let images = records
            .iter()
            .map(|r| fingerprint(Path::new(r["image"].as_str().ok_or("小节图片缺失")?)))
            .collect::<Result<Vec<_>>>()?;
        let context = json!({"native_context":1,"layout":fingerprint(&layout_path)?,"info":fingerprint(&info_path)?,"images":images,"models":self.models.catalog_identity(),"policy":self.models.policy,"capabilities":self.models.capabilities,"retry":state["ocr_task"]["measures"],"initial":initial,"options":[selected_device,8192,config.max_tokens,config.max_tokens_ceiling,config.maximum_attempts]});
        let context_file = output.join("recognition_context.json");
        let context_matches = view::read_json(&context_file)
            .ok()
            .map(|mut previous| {
                // Relocating downloads does not change the model or completed measures.
                previous["models"] = Models::content_identity(&previous["models"]);
                previous
            })
            .as_ref()
            == Some(&context);
        if resume
            && !context_matches
            && output
                .join("recognition.jsonl")
                .metadata()
                .is_ok_and(|m| m.len() > 0)
        {
            return Err("任务输入、模型或设备已改变，不能续跑；请重新开始识别。旧结果保留".into());
        }
        if !context_matches {
            edit::json_file(&context_file, &context)?;
        }
        self.models.ensure(allow_download, progress)?;
        initialize_runtime(self.native.ort.as_ref().ok_or("缺少原生ONNX组件")?)?;
        let states_file = output.join("state_predictions.json");
        let states = if resume && context_matches && states_file.is_file() {
            let states = view::read_json(&states_file)?;
            recognition::resolve_score_states(&mut records, &states)?;
            states
        } else {
            let mut reader = SignatureReader::new(&self.models.signature())?;
            let states = recognition::read_score_states(&mut reader, &mut records, &|| {
                progress.cancelled()
            })?;
            edit::json_file(&states_file, &states)?;
            states
        };
        let _ = states;
        let journal = output.join("recognition.jsonl");
        let mut completed = if resume {
            read_journal(&journal)?
        } else {
            BTreeMap::new()
        };
        if let Some(selected) = &retry {
            for row in initial["records"].as_array().ok_or("缺少已接受结果")? {
                let number = row["measure_number"].as_u64().ok_or("无效小节编号")? as usize;
                if !selected.contains(&number) {
                    completed.insert(number, row.clone());
                }
            }
        }
        self.with_engine(allow_download,progress,|engine|{
            if self.device()!=selected_device{return Err("识别设备在任务间改变，请重新开始本轮识别；旧结果保留".into());}
            let mut log=OpenOptions::new().create(true).append(true).open(&journal).map_err(|e|e.to_string())?;
            for record in &records{progress.checkpoint()?;let number=record["measure_number"].as_u64().ok_or("小节编号缺失")?as usize;
                if !completed.contains_key(&number){let result=recognition::recognize_measure(engine,record,&config,&||progress.cancelled())?;progress.checkpoint()?;serde_json::to_writer(&mut log,&result.record).map_err(|e|e.to_string())?;log.write_all(b"\n").map_err(|e|e.to_string())?;log.flush().map_err(|e|e.to_string())?;completed.insert(number,result.record);}
                progress.update(json!({"done":completed.len(),"total":records.len(),"message":format!("正在识别第 {} / {} 小节",completed.len(),records.len())}))?;
            }Ok(())
        })?;
        let mut finished = records
            .iter()
            .map(|r| {
                completed
                    .remove(&(r["measure_number"].as_u64().unwrap() as usize))
                    .ok_or("识别结果不完整".into())
            })
            .collect::<Result<Vec<_>>>()?;
        finalize_selected(&mut finished, &mut info, retry.as_ref())?;
        if retry.is_some() {
            // A local retry is not permission to change the accepted score's
            // shared instrument/tuning/tempo context beneath untouched records.
            for key in [
                "title",
                "artist",
                "instrument",
                "midi_program",
                "tuning_used",
                "capo",
                "transpose",
                "parts",
                "document_metadata",
            ] {
                if let Some(value) = initial.get(key) {
                    info[key] = value.clone();
                }
            }
        }
        let mut result = info;
        result["records"] = json!(finished);
        result["layout"] = json!(layout_path);
        result["info"] = json!(info_path);
        result["mode"] = layout["mode"].clone();
        result["recognition_log"] = json!(journal);
        let manifest = edit::save_recognition(&output, &result)?;
        progress.checkpoint()?;
        state = load_session(root, sid).map_err(|e| e.to_string())?;
        state["recognition"] = json!(manifest);
        state["export"] = Value::Null;
        state.as_object_mut().unwrap().remove("ocr_task");
        bump(&mut state)?;
        write_session_atomic(root, sid, &state).map_err(|e| e.to_string())?;
        Ok(())
    }
    pub fn process(&self, root: &Path, sid: &str, body: &Value, progress: &Progress) -> Result<()> {
        let download = body["allow_download"] == true;
        let initial = load_session(root, sid).map_err(|e| e.to_string())?;
        let requested = body["mode"]
            .as_str()
            .or(initial["mode_setting"].as_str())
            .unwrap_or("auto");
        for stage in ["layout", "info", "recognition"] {
            progress.checkpoint()?;
            let state = load_session(root, sid).map_err(|e| e.to_string())?;
            if state[stage].is_string()
                && !(stage == "recognition" && state["ocr_task"].is_object())
            {
                continue;
            }
            match stage {
                "layout" => self.detect(root, sid, requested, download, progress)?,
                "info" => self.information(root, sid, download, progress)?,
                _ => {
                    let mut options = body.clone();
                    options["resume"] = json!(state["ocr_task"].is_object());
                    self.recognize(root, sid, &options, progress)?;
                }
            }
        }
        Ok(())
    }
}
fn bump(state: &mut Value) -> Result<()> {
    state["revision"] = json!(state["revision"]
        .as_u64()
        .ok_or("Invalid project revision")?
        .checked_add(1)
        .ok_or("Project revision overflow")?);
    Ok(())
}
fn stage(directory: &Path, state: &Value, key: &str, kind: &str) -> Result<(PathBuf, Value)> {
    let path = Path::new(state[key].as_str().ok_or(format!("请先完成{key}步骤"))?)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if !path.starts_with(directory) {
        return Err("阶段路径超出项目".into());
    }
    let value = view::read_json(&path)?;
    if value["schema_version"] != "1.0" || value["stage"] != kind {
        return Err("无效的阶段清单".into());
    }
    Ok((path, value))
}
fn fingerprint(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    std::io::copy(&mut file, &mut HashWriter(&mut hash)).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", hash.finalize()))
}
struct HashWriter<'a>(&'a mut Sha256);
impl Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn read_journal(path: &Path) -> Result<BTreeMap<usize, Value>> {
    let mut result = BTreeMap::new();
    if !path.is_file() {
        return Ok(result);
    }
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let end = bytes
        .iter()
        .rposition(|b| *b == b'\n')
        .map(|n| n + 1)
        .unwrap_or(0);
    for line in bytes[..end]
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
    {
        let row: Value = serde_json::from_slice(line).map_err(|e| e.to_string())?;
        let number = row["measure_number"]
            .as_u64()
            .ok_or("Invalid checkpoint row")? as usize;
        scorelib::score::parse_measure_target(
            row["target"].as_str().ok_or("Invalid checkpoint music")?,
        )?;
        result.insert(number, row);
    }
    if end < bytes.len() {
        OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(|e| e.to_string())?
            .set_len(end as u64)
            .map_err(|e| e.to_string())?;
    }
    Ok(result)
}

fn finalize_selected(
    records: &mut [Value],
    information: &mut Value,
    retry: Option<&HashSet<usize>>,
) -> Result<()> {
    let preserved: BTreeMap<usize, Value> = records
        .iter()
        .filter_map(|row| {
            let number = row["measure_number"].as_u64()? as usize;
            if retry.is_some_and(|selected| !selected.contains(&number)) {
                Some((number, row.clone()))
            } else {
                None
            }
        })
        .collect();
    recognition::finalize_records(records, information)?;
    // A partial retry may use accepted neighbours as context, but never writes
    // their tempo, annotations, tuning, playback fields, or review decisions.
    for row in records {
        if let Some(original) = row["measure_number"]
            .as_u64()
            .and_then(|n| preserved.get(&(n as usize)))
        {
            *row = original.clone();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retry_preserves_every_unselected_manual_field_and_recovers_journal_tail() {
        let workspace = tempfile::tempdir().unwrap();
        let demo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../ui/workbench/examples/Harbor-Light-synthetic-project.zip");
        let project =
            crate::project::import_project(demo, workspace.path(), &Default::default()).unwrap();
        let source =
            view::read_json(Path::new(project.session["recognition"].as_str().unwrap())).unwrap();
        let mut information =
            view::read_json(Path::new(project.session["info"].as_str().unwrap())).unwrap();
        let mut rows = source["records"].as_array().unwrap().clone();
        let mut parsed =
            scorelib::score::parse_measure_target(rows[0]["target"].as_str().unwrap()).unwrap();
        parsed["tempo_quarter"] = json!(132);
        rows[0]["target"] = json!(scorelib::score::format_measure_target(
            &parsed,
            rows[0]["mode"].as_str().unwrap(),
            true
        )
        .unwrap());
        rows[0]["manually_edited"] = json!(true);
        rows[0]["reviewed"] = json!(true);
        rows[0]["chord_annotations"] = json!([]);
        let accepted = rows[0].clone();
        information["document_metadata"]["tempo_quarter"] = json!(120);
        finalize_selected(&mut rows, &mut information, Some(&HashSet::from([2]))).unwrap();
        assert_eq!(rows[0], accepted);
        let journal = workspace.path().join("journal.jsonl");
        let mut bytes = serde_json::to_vec(&accepted).unwrap();
        bytes.extend_from_slice(b"\n{\"measure_number\":");
        fs::write(&journal, bytes).unwrap();
        let recovered = read_journal(&journal).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[&1], accepted);
        assert!(fs::read(&journal).unwrap().ends_with(b"\n"));
    }
}
