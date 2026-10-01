//! Native desktop application service and local workbench entry point.
//! Uses the existing workbench asset/API shape; no Python process or dependency.
mod assets;
mod edit;
mod layout;
mod models;
mod pipeline;
mod tasks;
mod uploads;
mod view;
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Path, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use guitarocr_native_core::project::{import_project, ImportOptions};
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
async fn local_only(request: Request, next: Next) -> Response {
    let authority = request
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let host = authority.split(':').next().unwrap_or("");
    if host != "127.0.0.1" && host != "localhost" {
        return ApiError(StatusCode::FORBIDDEN, "不允许的主机地址".into()).into_response();
    }
    if let Some(origin) = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
    {
        if origin != format!("http://{authority}") {
            return ApiError(StatusCode::FORBIDDEN, "不允许跨站访问项目".into()).into_response();
        }
    }
    next.run(request).await
}
async fn config(State(app): State<AppState>) -> Json<Value> {
    Json(
        json!({"device":app.pipeline.device()["device"],"max_upload_mb":200,"max_pages":100,
        "inference_enabled":app.pipeline.available(),"model_ready":app.models.ready(),"model_cached":app.models.cached(),
        "layout_ready":app.native.ort.is_some()&&app.models.ready(),"layout_cached":app.models.layout().is_file(),
        "pdf_enabled":app.native.pdfium.is_some(),
        "native":true,"preview":true,"editing_enabled":true,"native_runtime_error":app.native_error,"model_download_bytes":app.models.total_bytes(),"requires_model_confirmation":!app.models.ready()}),
    )
}

async fn session(State(app): State<AppState>, Path(sid): Path<String>) -> ApiResult {
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
async fn sessions(State(app): State<AppState>) -> ApiResult {
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
async fn restore(State(app): State<AppState>, mut multipart: Multipart) -> ApiResult {
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
    State(app): State<AppState>,
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
    State(app): State<AppState>,
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
    let saved = guitarocr_native_core::project::load_session(&app.projects, &sid).map_err(bad)?;
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
    State(app): State<AppState>,
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
    State(app): State<AppState>,
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
    State(app): State<AppState>,
    Path(sid): Path<String>,
    headers: axum::http::HeaderMap,
) -> ApiResult {
    let directory = view::directory(&app.projects, &sid).map_err(bad)?;
    let _guard = app.mutation.lock().map_err(|_| bad("项目状态不可用"))?;
    app.tasks
        .idle(&sid)
        .map_err(|e| ApiError(StatusCode::CONFLICT, e))?;
    if directory.join("session.json").is_file() {
        let saved =
            guitarocr_native_core::project::load_session(&app.projects, &sid).map_err(bad)?;
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
    State(app): State<AppState>,
    Path(sid): Path<String>,
) -> Result<Response, ApiError> {
    let saved = guitarocr_native_core::project::load_session(&app.projects, &sid).map_err(bad)?;
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
        guitarocr_native_core::score::display_score_text(&text),
    )
        .into_response())
}
async fn cancel(State(app): State<AppState>, Path(sid): Path<String>) -> ApiResult {
    view::directory(&app.projects, &sid).map_err(bad)?;
    app.tasks.cancel(&sid).map(Json).map_err(bad)
}
async fn correct(
    State(app): State<AppState>,
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
    State(app): State<AppState>,
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
    State(app): State<AppState>,
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
        let saved = guitarocr_native_core::project::load_session(&root, &id).map_err(bad)?;
        let directory = root.join(".archives");
        fs::create_dir_all(&directory).map_err(bad)?;
        let path = directory.join(format!("{}-{}.zip", id, saved["revision"]));
        guitarocr_native_core::project::export_project(&root, &id, &path).map_err(bad)
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
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut projects = None;
    let mut assets = None;
    let mut port = 0u16;
    let mut model_root = std::env::var_os("GUITAROCR_MODEL_CACHE").map(PathBuf::from);
    let mut resources = std::env::var_os("GUITAROCR_NATIVE_RESOURCES").map(PathBuf::from);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--projects" => projects = args.next().map(PathBuf::from),
            "--assets" => assets = args.next().map(PathBuf::from),
            "--models" => model_root = args.next().map(PathBuf::from),
            "--resources" => resources = args.next().map(PathBuf::from),
            "--port" => port = args.next().ok_or("Missing --port value")?.parse()?,
            _ => {
                return Err(format!(
                    "Unknown argument {arg}; use --projects DIR --assets DIR [--port PORT]"
                )
                .into())
            }
        }
    }
    let projects = projects.ok_or("--projects DIR is required")?;
    let assets = assets.ok_or("--assets DIR is required")?.canonicalize()?;
    if !assets.join("index.html").is_file() {
        return Err("Workbench assets are missing".into());
    }
    fs::create_dir_all(&projects)?;
    let projects = projects.canonicalize()?;
    let tasks = tasks::Tasks::new(projects.clone())?;
    let (native, native_error) = match assets::NativeAssets::load(resources.as_deref()) {
        Ok(native) => (native, None),
        Err(error) => {
            eprintln!("本机识别组件未启用：{error}；项目编辑仍可用");
            (assets::NativeAssets::default(), Some(error))
        }
    };
    let models = Arc::new(models::Models::new(
        projects.parent().unwrap_or(&projects),
        model_root,
    )?);
    let pipeline = Arc::new(pipeline::Pipeline::new(native.clone(), models.clone()));
    let state = Arc::new(App {
        projects,
        tasks,
        native,
        native_error,
        models,
        pipeline,
        mutation: Mutex::new(()),
    });
    let app = Router::new()
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
        )
        .nest_service("/static", ServeDir::new(&assets))
        .route_service("/", ServeFile::new(assets.join("index.html")))
        .layer(DefaultBodyLimit::max(251 * 1024 * 1024))
        .layer(middleware::from_fn(local_only))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!("GUITAROCR_READY http://{}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
