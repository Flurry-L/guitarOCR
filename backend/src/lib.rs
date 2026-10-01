//! Shared product backend for local and hosted operation.
mod accounts;
pub mod assets;
pub mod auxiliary_onnx;
pub mod edit;
pub mod image_boundary;
pub mod image_transforms;
pub mod layout;
pub mod layout_postprocess;
pub mod models;
pub mod ottava_geometry;
pub mod pipeline;
pub mod pixel_refinement;
pub mod project;
pub mod recognition;
pub mod score_grid;
pub mod score_structure;
pub mod staff_classifier;
pub mod staff_geometry;
pub mod tasks;
pub mod uploads;
pub mod view;
use crate::project::{import_project, ImportOptions};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Path, Request},
    http::{header, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Extension, Json, Router,
};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::UNIX_EPOCH,
};
use tower_http::services::{ServeDir, ServeFile};

struct App {
    projects: PathBuf,
    mutation: Mutex<()>,
    tasks: Arc<tasks::Tasks>,
    native: assets::NativeAssets,
    native_error: Option<String>,
    models: Arc<models::Models>,
    pipeline: Arc<pipeline::Pipeline>,
    accounts: Option<Arc<accounts::Accounts>>,
    tenants: Mutex<std::collections::HashMap<String, AppState>>,
}
type AppState = Arc<App>;
type ApiResult = Result<Json<Value>, ApiError>;
struct ApiError(StatusCode, String);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"detail": self.1}))).into_response()
    }
}
fn bad(error: impl ToString) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, error.to_string())
}
async fn config(Extension(app): Extension<AppState>) -> Json<Value> {
    Json(
        json!({"device":app.pipeline.device()["device"],"max_upload_mb":200,"max_pages":100,
        "inference_enabled":app.pipeline.available(),"model_ready":app.models.ready(),"model_cached":app.models.cached(),
        "layout_ready":app.native.ort.is_some()&&app.models.ready(),"layout_cached":app.models.layout().is_file(),
        "pdf_enabled":app.native.pdfium.is_some(),
        "native":true,"server":app.accounts.is_some(),"registration":app.accounts.as_ref().is_some_and(|a|a.registration()),"gpu_available":app.pipeline.available(),"editing_enabled":true,"native_runtime_error":app.native_error,"model_download_bytes":app.models.total_bytes(),"requires_model_confirmation":!app.models.ready()}),
    )
}

async fn session(Extension(app): Extension<AppState>, Path(sid): Path<String>) -> ApiResult {
    let dir = view::directory(&app.projects, &sid).map_err(bad)?;
    let mut state = if dir.join("session.json").is_file() {
        view::project(&app.projects, &sid).map_err(bad)?
    } else if app.tasks.snapshot(&sid).is_some() {
        json!({"id":sid})
    } else {
        return Err(ApiError(StatusCode::NOT_FOUND, "项目不存在".into()));
    };
    state["job"] = app.tasks.snapshot(&sid).unwrap_or(Value::Null);
    Ok(Json(state))
}
async fn sessions(Extension(app): Extension<AppState>) -> ApiResult {
    let mut rows = Vec::new();
    for entry in fs::read_dir(&app.projects).map_err(bad)? {
        let entry = entry.map_err(bad)?;
        let sid = entry.file_name().to_string_lossy().into_owned();
        if view::directory(&app.projects, &sid).is_err() {
            continue;
        }
        let path = entry.path().join("session.json");
        let Ok(saved) = view::read_json(&path) else {
            continue;
        };
        let Ok(project) = view::project(&app.projects, &sid) else {
            continue;
        };
        let title = project["metadata"]["title"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| {
                saved["input_names"]
                    .as_array()
                    .map(|rows| {
                        rows.iter()
                            .filter_map(Value::as_str)
                            .collect::<Vec<_>>()
                            .join("、")
                    })
                    .unwrap_or_default()
            });
        let updated = path
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|v| v.as_secs_f64())
            .unwrap_or(0.);
        rows.push(json!({"id":sid, "title":title, "pages":saved["pages"].as_array().map(Vec::len).unwrap_or(0),
            "stage":if saved["export"].is_string() {"已导出"} else if saved["recognition"].is_string() {"待校对"} else {"编辑中"}, "updated":updated, "job":app.tasks.snapshot(&sid)}));
    }
    let known = rows
        .iter()
        .filter_map(|r| r["id"].as_str())
        .map(str::to_owned)
        .collect::<std::collections::HashSet<_>>();
    for (sid, job) in app.tasks.snapshots() {
        if !known.contains(&sid) {
            rows.push(json!({"id":sid,"title":job["title"].as_str().unwrap_or("正在导入的乐谱"),"pages":0,"stage":"待导入","updated":job["created"],"job":job}));
        }
    }
    rows.sort_by(|a, b| {
        b["updated"]
            .as_f64()
            .partial_cmp(&a["updated"].as_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(Json(json!(rows)))
}
async fn restore(Extension(app): Extension<AppState>, mut multipart: Multipart) -> ApiResult {
    app.check_quota()?;
    let mut found = false;
    let mut uploaded = tempfile::NamedTempFile::new_in(&app.projects).map_err(bad)?;
    while let Some(mut field) = multipart.next_field().await.map_err(bad)? {
        if field.name() != Some("file") || found {
            return Err(bad("请选择一个项目 ZIP"));
        }
        found = true;
        let mut size = 0usize;
        while let Some(chunk) = field.chunk().await.map_err(bad)? {
            size += chunk.len();
            if size > 250 * 1024 * 1024 {
                return Err(ApiError(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "项目 ZIP 最大 250 MB".into(),
                ));
            }
            uploaded.write_all(&chunk).map_err(bad)?;
        }
    }
    if !found {
        return Err(bad("请选择项目 ZIP"));
    }
    uploaded.flush().map_err(bad)?;
    let root = app.projects.clone();
    let imported = tokio::task::spawn_blocking(move || {
        import_project(uploaded.path(), root, &ImportOptions::default())
    })
    .await
    .map_err(bad)?
    .map_err(bad)?;
    view::project(&app.projects, &imported.session_id)
        .map(Json)
        .map_err(bad)
}
async fn asset(
    Extension(app): Extension<AppState>,
    Path((sid, name)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let root = view::directory(&app.projects, &sid)
        .map_err(bad)?
        .canonicalize()
        .map_err(bad)?;
    let path = root
        .join(name)
        .canonicalize()
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "文件不存在".into()))?;
    if !path.starts_with(&root) || !path.is_file() {
        return Err(ApiError(StatusCode::FORBIDDEN, "无效的资源路径".into()));
    }
    // ServeFile streams the body instead of loading potentially large PDFs/ZIPs.
    use tower::ServiceExt;
    let response = ServeFile::new(path)
        .oneshot(Request::new(Body::empty()))
        .await
        .map_err(bad)?;
    Ok(response.into_response())
}
async fn staged_task(
    Extension(app): Extension<AppState>,
    Path((sid, action)): Path<(String, String)>,
    headers: axum::http::HeaderMap,
    body: Option<Json<Value>>,
) -> ApiResult {
    if !matches!(
        action.as_str(),
        "process" | "detect" | "information" | "recognize"
    ) {
        return Err(ApiError(StatusCode::NOT_FOUND, "未知的处理步骤".into()));
    }
    let body = body.map(|Json(v)| v).unwrap_or(json!({}));
    if !body.is_object() {
        return Err(bad("处理选项必须是对象"));
    }
    if body["source"] == "geometry" {
        return Err(bad(
            "原生客户端使用ONNX和图像几何，请选择自动或图像来源；PDF矢量专用路径仍由研究工具提供",
        ));
    }
    if !app.pipeline.available() {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "原生识别组件未完整准备，请使用完整原生安装包".into(),
        ));
    }
    view::directory(&app.projects, &sid).map_err(bad)?;
    let _guard = app.mutation.lock().map_err(|_| bad("项目状态不可用"))?;
    app.tasks
        .idle(&sid)
        .map_err(|e| ApiError(StatusCode::CONFLICT, e))?;
    let saved = crate::project::load_session(&app.projects, &sid).map_err(bad)?;
    edit::revision(
        &saved,
        headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok()),
    )
    .map_err(|(code, message)| ApiError(StatusCode::from_u16(code).unwrap(), message))?;
    let pipeline = app.pipeline.clone();
    let root = app.projects.clone();
    let id = sid.clone();
    let which = action.clone();
    let result = app
        .tasks
        .submit(
            sid,
            "正在准备本机处理",
            if action == "process" { "full" } else { &action },
            move |progress| match which.as_str() {
                "process" => pipeline.process(&root, &id, &body, &progress),
                "detect" => pipeline.detect(
                    &root,
                    &id,
                    body["mode"].as_str().unwrap_or("auto"),
                    body["allow_download"] == true,
                    &progress,
                ),
                "information" => {
                    pipeline.information(&root, &id, body["allow_download"] == true, &progress)
                }
                _ => pipeline.recognize(&root, &id, &body, &progress),
            },
        )
        .map_err(bad)?;
    Ok(Json(result))
}
async fn metadata(
    Extension(app): Extension<AppState>,
    Path(sid): Path<String>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult {
    view::directory(&app.projects, &sid).map_err(bad)?;
    let _guard = app.mutation.lock().map_err(|_| bad("项目状态不可用"))?;
    app.tasks
        .idle(&sid)
        .map_err(|e| ApiError(StatusCode::CONFLICT, e))?;
    edit::metadata(
        &app.projects,
        &sid,
        &body,
        headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok()),
    )
    .map_err(|(code, message)| ApiError(StatusCode::from_u16(code).unwrap(), message))?;
    app.tasks.clear(&sid).map_err(bad)?;
    view::project(&app.projects, &sid).map(Json).map_err(bad)
}
async fn boxes(
    Extension(app): Extension<AppState>,
    Path(sid): Path<String>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult {
    view::directory(&app.projects, &sid).map_err(bad)?;
    let _guard = app.mutation.lock().map_err(|_| bad("项目状态不可用"))?;
    app.tasks
        .idle(&sid)
        .map_err(|e| ApiError(StatusCode::CONFLICT, e))?;
    layout::save_boxes(
        &app.projects,
        &sid,
        &body,
        headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok()),
    )
    .map_err(|(code, message)| ApiError(StatusCode::from_u16(code).unwrap(), message))?;
    app.tasks.clear(&sid).map_err(bad)?;
    view::project(&app.projects, &sid).map(Json).map_err(bad)
}
async fn delete(
    Extension(app): Extension<AppState>,
    Path(sid): Path<String>,
    headers: axum::http::HeaderMap,
) -> ApiResult {
    let directory = view::directory(&app.projects, &sid).map_err(bad)?;
    let _guard = app.mutation.lock().map_err(|_| bad("项目状态不可用"))?;
    app.tasks
        .idle(&sid)
        .map_err(|e| ApiError(StatusCode::CONFLICT, e))?;
    if directory.join("session.json").is_file() {
        let saved = crate::project::load_session(&app.projects, &sid).map_err(bad)?;
        edit::revision(
            &saved,
            headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok()),
        )
        .map_err(|(code, message)| ApiError(StatusCode::from_u16(code).unwrap(), message))?;
    }
    let trash = app.projects.join(".trash");
    fs::create_dir_all(&trash).map_err(bad)?;
    let target = tempfile::Builder::new()
        .prefix(&format!("{sid}-"))
        .tempdir_in(&trash)
        .map_err(bad)?;
    fs::rename(&directory, target.path().join("project")).map_err(bad)?;
    let _ = target.keep();
    app.tasks.clear(&sid).map_err(bad)?;
    Ok(Json(json!({"deleted":true})))
}
async fn score_text(
    Extension(app): Extension<AppState>,
    Path(sid): Path<String>,
) -> Result<Response, ApiError> {
    let saved = crate::project::load_session(&app.projects, &sid).map_err(bad)?;
    let source = view::read_json(std::path::Path::new(
        saved["recognition"]
            .as_str()
            .ok_or_else(|| bad("请先识别小节"))?,
    ))
    .map_err(bad)?;
    let text = source["records"]
        .as_array()
        .ok_or_else(|| bad("识别结果缺少小节"))?
        .iter()
        .filter_map(|r| r["target"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    Ok((
        [
            (header::CONTENT_TYPE, "text/plain; charset=utf-8"),
            (
                header::CONTENT_DISPOSITION,
                "attachment; filename=score.txt",
            ),
            (header::CACHE_CONTROL, "no-store"),
        ],
        scorelib::score::display_score_text(&text),
    )
        .into_response())
}
async fn cancel(Extension(app): Extension<AppState>, Path(sid): Path<String>) -> ApiResult {
    view::directory(&app.projects, &sid).map_err(bad)?;
    app.tasks.cancel(&sid).map(Json).map_err(bad)
}
async fn correct(
    Extension(app): Extension<AppState>,
    Path((sid, number)): Path<(String, usize)>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> ApiResult {
    view::directory(&app.projects, &sid).map_err(bad)?;
    let _guard = app.mutation.lock().map_err(|_| bad("项目状态不可用"))?;
    app.tasks
        .idle(&sid)
        .map_err(|e| ApiError(StatusCode::CONFLICT, e))?;
    edit::correct(
        &app.projects,
        &sid,
        number,
        &body,
        headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok()),
    )
    .map_err(|(code, error)| {
        ApiError(
            StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_REQUEST),
            error,
        )
    })?;
    app.tasks.clear(&sid).map_err(bad)?;
    view::project(&app.projects, &sid).map(Json).map_err(bad)
}
async fn export(
    Extension(app): Extension<AppState>,
    Path(sid): Path<String>,
    headers: axum::http::HeaderMap,
) -> ApiResult {
    view::directory(&app.projects, &sid).map_err(bad)?;
    let _guard = app.mutation.lock().map_err(|_| bad("项目状态不可用"))?;
    app.tasks
        .idle(&sid)
        .map_err(|e| ApiError(StatusCode::CONFLICT, e))?;
    edit::export(
        &app.projects,
        &sid,
        headers.get(header::IF_MATCH).and_then(|v| v.to_str().ok()),
    )
    .map_err(|(code, error)| {
        ApiError(
            StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_REQUEST),
            error,
        )
    })?;
    app.tasks.clear(&sid).map_err(bad)?;
    view::project(&app.projects, &sid).map(Json).map_err(bad)
}
async fn archive(
    Extension(app): Extension<AppState>,
    Path(sid): Path<String>,
) -> Result<Response, ApiError> {
    view::directory(&app.projects, &sid).map_err(bad)?;
    let root = app.projects.clone();
    let id = sid.clone();
    let path = tokio::task::spawn_blocking(move || {
        let _guard = app.mutation.lock().map_err(|_| bad("项目状态不可用"))?;
        app.tasks
            .idle(&id)
            .map_err(|message| ApiError(StatusCode::CONFLICT, message))?;
        let saved = crate::project::load_session(&root, &id).map_err(bad)?;
        let directory = root.join(".archives");
        fs::create_dir_all(&directory).map_err(bad)?;
        let path = directory.join(format!("{}-{}.zip", id, saved["revision"]));
        crate::project::export_project(&root, &id, &path).map_err(bad)
    })
    .await
    .map_err(bad)??;
    use tower::ServiceExt;
    let mut response = ServeFile::new(path)
        .oneshot(Request::new(Body::empty()))
        .await
        .map_err(bad)?
        .into_response();
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        format!(
            "attachment; filename=\"guitarocr-project-{}.zip\"",
            &sid[..8]
        )
        .parse()
        .map_err(bad)?,
    );
    Ok(response)
}
impl App {
    fn for_user(&self, id: &str) -> Result<AppState, String> {
        if id.len() != 64 || !id.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err("无效的账号编号".into());
        }
        let mut tenants = self.tenants.lock().map_err(|_| "项目服务忙")?;
        if let Some(app) = tenants.get(id) {
            return Ok(app.clone());
        }
        let projects = self.projects.join("users").join(id);
        fs::create_dir_all(&projects).map_err(|e| e.to_string())?;
        let app = Arc::new(App {
            tasks: tasks::Tasks::new(projects.clone())?,
            projects,
            mutation: Mutex::new(()),
            native: self.native.clone(),
            native_error: self.native_error.clone(),
            models: self.models.clone(),
            pipeline: self.pipeline.clone(),
            accounts: self.accounts.clone(),
            tenants: Mutex::new(Default::default()),
        });
        tenants.insert(id.to_string(), app.clone());
        Ok(app)
    }
    fn usage(&self) -> Value {
        fn bytes(path: &std::path::Path) -> u64 {
            fs::read_dir(path)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| {
                    let Ok(meta) = e.metadata() else { return 0 };
                    if e.file_type().is_ok_and(|t| t.is_symlink()) {
                        0
                    } else if meta.is_dir() {
                        bytes(&e.path())
                    } else {
                        meta.len()
                    }
                })
                .sum()
        }
        let mut projects = 0;
        let mut pages = 0;
        for entry in fs::read_dir(&self.projects).into_iter().flatten().flatten() {
            if view::directory(&self.projects, &entry.file_name().to_string_lossy()).is_err() {
                continue;
            }
            projects += 1;
            pages += view::read_json(&entry.path().join("session.json"))
                .ok()
                .and_then(|v| v["pages"].as_array().map(Vec::len))
                .unwrap_or(0);
        }
        let jobs = self.tasks.snapshots();
        let seconds: f64 = jobs
            .iter()
            .map(|(_, j)| {
                (j["finished"].as_f64().unwrap_or(0.) - j["created"].as_f64().unwrap_or(0.)).max(0.)
            })
            .sum();
        json!({"projects":projects,"pages":pages,"bytes":bytes(&self.projects),"engines":[{"engine":"native","jobs":jobs.len(),"seconds":seconds}]})
    }
    fn check_quota(&self) -> Result<(), ApiError> {
        if self.accounts.is_some() {
            let usage = self.usage();
            if usage["projects"].as_u64().unwrap_or(0) >= 200
                || usage["bytes"].as_u64().unwrap_or(0) >= 10 * 1024 * 1024 * 1024
            {
                return Err(ApiError(
                    StatusCode::CONFLICT,
                    "账号空间已满，请先删除不需要的项目".into(),
                ));
            }
        }
        Ok(())
    }
}
async fn usage(Extension(app): Extension<AppState>) -> Json<Value> {
    Json(app.usage())
}

/// Native host entry; local and server modes use identical project handlers.
pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut projects = PathBuf::from("output/projects");
    let mut ui = PathBuf::from("ui/workbench");
    let mut port = 0u16;
    let mut bind = "127.0.0.1".to_string();
    let mut server = false;
    let mut api_only = false;
    let mut public_url = None;
    let mut admin = None;
    let mut slots = None;
    let mut model_root = std::env::var_os("GUITAROCR_MODEL_CACHE").map(PathBuf::from);
    let mut resources = std::env::var_os("GUITAROCR_NATIVE_RESOURCES").map(PathBuf::from);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--projects" => {
                projects = PathBuf::from(args.next().ok_or("Missing --projects value")?)
            }
            "--assets" => ui = PathBuf::from(args.next().ok_or("Missing --assets value")?),
            "--models" => {
                model_root = Some(PathBuf::from(args.next().ok_or("Missing --models value")?))
            }
            "--resources" => {
                resources = Some(PathBuf::from(
                    args.next().ok_or("Missing --resources value")?,
                ))
            }
            "--port" => port = args.next().ok_or("Missing --port value")?.parse()?,
            "--bind" => bind = args.next().ok_or("Missing --bind value")?,
            "--public-url" => public_url = args.next(),
            "--create-admin" => admin = args.next(),
            "--slots" => {
                slots = Some(
                    args.next()
                        .ok_or("Missing --slots value")?
                        .parse::<usize>()?,
                )
            }
            "--server" => server = true,
            "--api-only" => api_only = true,
            "--help" | "-h" => {
                println!("guitarocr-backend [--projects DIR] [--assets ui/workbench] [--resources DIR] [--models DIR] [--port PORT] [--slots 1..4]\n  Local mode binds loopback. Hosted mode: --server --bind ADDRESS --public-url URL [--api-only]\n  Create an admin: --projects DIR --create-admin USER (GUITAROCR_ADMIN_PASSWORD from environment)");
                return Ok(());
            }
            _ => return Err(format!("Unknown argument {arg}; use --help").into()),
        }
    }
    let address: std::net::IpAddr = bind.parse()?;
    if !server && !address.is_loopback() {
        return Err("Use --server and --public-url for network access".into());
    }
    let slots = slots.unwrap_or(if server { 4 } else { 1 });
    if !(1..=4).contains(&slots) {
        return Err("--slots must be between 1 and 4".into());
    }
    fs::create_dir_all(&projects)?;
    let projects = projects.canonicalize()?;
    let accounts = if server || admin.is_some() {
        let origin = public_url.unwrap_or_default();
        if server {
            let url = reqwest::Url::parse(&origin)
                .map_err(|_| "Hosted mode requires --public-url http(s)://host[:port]")?;
            if !matches!(url.scheme(), "http" | "https")
                || url.origin().ascii_serialization() != origin
            {
                return Err(
                    "--public-url must be an HTTP(S) origin without a trailing slash".into(),
                );
            }
        }
        Some(accounts::Accounts::open(
            &projects.join("accounts.sqlite3"),
            origin,
        )?)
    } else {
        None
    };
    if let Some(name) = admin {
        if !(3..=32).contains(&name.len())
            || !name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        {
            return Err("Invalid admin username".into());
        }
        let password = std::env::var("GUITAROCR_ADMIN_PASSWORD")
            .map_err(|_| "Set GUITAROCR_ADMIN_PASSWORD to create the admin")?;
        let accounts = accounts.unwrap();
        tokio::task::spawn_blocking(move || accounts.add_user(&name, &password, true)).await??;
        println!("管理员已创建");
        return Ok(());
    }
    if accounts
        .as_ref()
        .is_some_and(|a| !a.has_admin().unwrap_or(false))
    {
        return Err("Create an administrator first with --create-admin USER".into());
    }
    let tasks = tasks::Tasks::new(projects.clone())?;
    let (native, native_error) = match assets::NativeAssets::load(resources.as_deref()) {
        Ok(native) => (native, None),
        Err(error) => {
            eprintln!("识别组件未启用：{error}");
            (assets::NativeAssets::default(), Some(error))
        }
    };
    let models = Arc::new(models::Models::new(
        projects.parent().unwrap_or(&projects),
        model_root,
    )?);
    let pipeline = Arc::new(pipeline::Pipeline::new(
        native.clone(),
        models.clone(),
        slots,
    ));
    let state = Arc::new(App {
        projects,
        tasks,
        native,
        native_error,
        models,
        pipeline,
        mutation: Mutex::new(()),
        accounts,
        tenants: Mutex::new(Default::default()),
    });
    // Recover task journals for every account, even before the owner logs in.
    if let Some(accounts) = &state.accounts {
        for user in accounts.users()? {
            state.for_user(user["id"].as_str().unwrap())?;
        }
    }
    let mut app = Router::new()
        .route("/health", get(|| async { Json(json!({"ok":true})) }))
        .route("/api/auth/register", post(accounts::register))
        .route("/api/auth/login", post(accounts::login))
        .route("/api/auth/me", get(accounts::me))
        .route("/api/auth/logout", post(accounts::logout))
        .route("/api/auth/password", put(accounts::password))
        .route("/api/usage", get(usage))
        .route("/api/admin/users", get(accounts::users))
        .route(
            "/api/admin/users/{id}",
            axum::routing::patch(accounts::update_user),
        )
        .route("/api/admin/status", get(accounts::status))
        .route("/api/admin/registration", put(accounts::registration))
        .route("/api/admin/jobs/{id}/cancel", post(accounts::cancel_job))
        .route("/api/config", get(config))
        .route("/api/sessions", get(sessions).post(uploads::upload))
        .route("/api/sessions/{sid}", get(session).delete(delete))
        .route("/api/sessions/{sid}/score.txt", get(score_text))
        .route("/api/sessions/{sid}/files/{*name}", get(asset))
        .route("/api/projects/import", post(restore))
        .route("/api/sessions/{sid}/{action}", post(staged_task))
        .route("/api/sessions/{sid}/cancel", post(cancel))
        .route(
            "/api/sessions/{sid}/metadata",
            put(metadata).layer(DefaultBodyLimit::max(128 * 1024)),
        )
        .route(
            "/api/sessions/{sid}/boxes",
            put(boxes).layer(DefaultBodyLimit::max(2 * 1024 * 1024)),
        )
        .route("/api/sessions/{sid}/archive", get(archive))
        .route("/api/sessions/{sid}/export", post(export))
        .route(
            "/api/sessions/{sid}/measures/{number}",
            put(correct).layer(DefaultBodyLimit::max(128 * 1024)),
        );
    if !api_only {
        let assets = ui.canonicalize()?;
        if !assets.join("index.html").is_file() {
            return Err("Workbench assets are missing".into());
        }
        app = app.nest_service("/static", ServeDir::new(&assets));
        if server {
            let accounts = assets.parent().ok_or("Invalid UI path")?.join("accounts");
            if !accounts.join("index.html").is_file() {
                return Err("Account UI assets are missing".into());
            }
            app = app
                .nest_service("/server-static", ServeDir::new(&accounts))
                .route_service("/", ServeFile::new(accounts.join("index.html")))
                .route_service("/workbench", ServeFile::new(assets.join("index.html")));
        } else {
            app = app.route_service("/", ServeFile::new(assets.join("index.html")));
        }
    }
    let app = app
        .layer(DefaultBodyLimit::max(251 * 1024 * 1024))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            accounts::access,
        ))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind((address, port)).await?;
    println!("GUITAROCR_READY http://{}", listener.local_addr()?);
    let result = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(async {
        #[cfg(unix)]
        {
            let mut terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("Cannot listen for service termination");
            tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
        }
        #[cfg(not(unix))]
        let _ = tokio::signal::ctrl_c().await;
    })
    .await;
    for (id, _) in state.tasks.snapshots() {
        let _ = state.tasks.cancel(&id);
    }
    if let Ok(tenants) = state.tenants.lock() {
        for tenant in tenants.values() {
            for (id, _) in tenant.tasks.snapshots() {
                let _ = tenant.tasks.cancel(&id);
            }
        }
    }
    state.pipeline.shutdown();
    result?;
    Ok(())
}
