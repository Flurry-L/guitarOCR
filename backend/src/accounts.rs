//! Hosted accounts and access control. The project handlers are shared with local mode.
use crate::{bad, ApiError, ApiResult, AppState};
use axum::{
    extract::{Path, Query, Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Extension, Json,
};
use rusqlite::{params, Connection, OptionalExtension};
use scrypt::{
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Scrypt,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    path::Path as FsPath,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

pub struct Accounts {
    db: Mutex<Connection>,
    pub origin: String,
    attempts: Mutex<HashMap<String, (u64, usize)>>,
}
#[derive(Clone)]
pub(crate) struct Identity {
    pub user: Value,
    csrf: String,
    token: String,
    expires: u64,
}
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn token() -> Result<String, String> {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|e| e.to_string())?;
    Ok(bytes.iter().map(|v| format!("{v:02x}")).collect())
}
fn digest(token: &str) -> String {
    format!("{:x}", Sha256::digest(token.as_bytes()))
}
fn password_hash(password: &str) -> Result<String, String> {
    if !(10..=128).contains(&password.chars().count()) {
        return Err("密码须为 10 至 128 个字符".into());
    }
    let mut salt = [0u8; 16];
    getrandom::fill(&mut salt).map_err(|e| e.to_string())?;
    let salt = SaltString::encode_b64(&salt).map_err(|e| e.to_string())?;
    Scrypt
        .hash_password(password.as_bytes(), &salt)
        .map(|v| v.to_string())
        .map_err(|e| e.to_string())
}
fn check_password(password: &str, encoded: &str) -> bool {
    password.len() <= 512
        && PasswordHash::new(encoded)
            .is_ok_and(|hash| Scrypt.verify_password(password.as_bytes(), &hash).is_ok())
}
fn username(value: &Value) -> Result<&str, ApiError> {
    value["username"]
        .as_str()
        .filter(|s| {
            (3..=32).contains(&s.len())
                && s.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        })
        .ok_or_else(|| bad("用户名须为 3 至 32 个字母、数字、下划线或短横线"))
}
impl Accounts {
    pub fn open(path: &FsPath, origin: String) -> Result<Arc<Self>, String> {
        let db = Connection::open(path).map_err(|e| e.to_string())?;
        db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, username TEXT UNIQUE COLLATE NOCASE NOT NULL, password TEXT NOT NULL, admin INTEGER NOT NULL DEFAULT 0, disabled INTEGER NOT NULL DEFAULT 0, created INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), csrf TEXT NOT NULL, expires INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);") .map_err(|e| e.to_string())?;
        Ok(Arc::new(Self {
            db: Mutex::new(db),
            origin,
            attempts: Mutex::new(HashMap::new()),
        }))
    }
    pub fn has_admin(&self) -> Result<bool, String> {
        self.db
            .lock()
            .map_err(|_| "Account database unavailable")?
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM users WHERE admin=1 AND disabled=0)",
                [],
                |r| r.get(0),
            )
            .map_err(|e| e.to_string())
    }
    pub fn add_user(&self, name: &str, password: &str, admin: bool) -> Result<Value, String> {
        let hash = password_hash(password)?;
        let id = token()?;
        let created = now();
        self.db
            .lock()
            .map_err(|_| "Account database unavailable")?
            .execute(
                "INSERT INTO users VALUES(?,?,?,?,0,?)",
                params![id, name, hash, admin, created],
            )
            .map_err(|e| {
                if e.sqlite_error_code() == Some(rusqlite::ErrorCode::ConstraintViolation) {
                    "用户名已被使用".to_string()
                } else {
                    e.to_string()
                }
            })?;
        Ok(json!({"id":id,"username":name,"admin":admin,"guest":false}))
    }
    pub fn registration(&self) -> bool {
        self.db
            .lock()
            .ok()
            .and_then(|db| {
                db.query_row(
                    "SELECT value FROM settings WHERE key='registration'",
                    [],
                    |r| r.get::<_, String>(0),
                )
                .ok()
            })
            .as_deref()
            == Some("true")
    }
    fn rate(&self, key: String, limit: usize) -> Result<(), ApiError> {
        let mut attempts = self.attempts.lock().map_err(|_| bad("账号服务忙"))?;
        attempts.retain(|_, (time, _)| now().saturating_sub(*time) < 900);
        if attempts.len() >= 4096 {
            return Err(ApiError(
                StatusCode::TOO_MANY_REQUESTS,
                "尝试过于频繁，请稍后重试".into(),
            ));
        }
        let row = attempts.entry(key).or_insert((now(), 0));
        row.1 += 1;
        if row.1 > limit {
            return Err(ApiError(
                StatusCode::TOO_MANY_REQUESTS,
                "尝试过于频繁，请稍后重试".into(),
            ));
        }
        Ok(())
    }
    fn authenticate(&self, request: &Request) -> Result<Option<Identity>, ApiError> {
        let cookie = request
            .headers()
            .get(header::COOKIE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let Some(token) = cookie
            .split(';')
            .filter_map(|v| v.trim().split_once('='))
            .find_map(|(k, v)| (k == "guitarocr-auth").then_some(v))
        else {
            return Ok(None);
        };
        let hashed = digest(token);
        self.db.lock().map_err(|_| bad("账号服务忙"))?.query_row("SELECT u.id,u.username,u.admin,s.csrf,s.expires FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=? AND s.expires>? AND u.disabled=0", params![hashed,now()], |r| Ok(Identity { user:json!({"id":r.get::<_,String>(0)?,"username":r.get::<_,String>(1)?,"admin":r.get::<_,bool>(2)?,"guest":false}), csrf:r.get(3)?, expires:r.get(4)?, token:hashed.clone() })).optional().map_err(bad)
    }
    fn login_response(&self, user: Value) -> Result<Response, ApiError> {
        let token = token().map_err(bad)?;
        let csrf = crate::accounts::token().map_err(bad)?;
        let expires = now() + 7 * 86400;
        let db = self.db.lock().map_err(|_| bad("账号服务忙"))?;
        db.execute("DELETE FROM sessions WHERE expires<?", [now()])
            .map_err(bad)?;
        db.execute(
            "INSERT INTO sessions VALUES(?,?,?,?)",
            params![digest(&token), user["id"].as_str(), csrf, expires],
        )
        .map_err(bad)?;
        let cookie = format!(
            "guitarocr-auth={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800{}",
            if self.origin.starts_with("https:") {
                "; Secure"
            } else {
                ""
            }
        );
        Ok((
            [(header::SET_COOKIE, cookie)],
            Json(json!({"user":user,"csrf":csrf,"expires":expires})),
        )
            .into_response())
    }
    pub fn users(&self) -> Result<Vec<Value>, String> {
        let db = self.db.lock().map_err(|_| "Account database unavailable")?;
        let mut statement = db
            .prepare("SELECT id,username,admin,disabled,created FROM users ORDER BY created")
            .map_err(|e| e.to_string())?;
        let values=statement.query_map([], |r|Ok(json!({"id":r.get::<_,String>(0)?,"username":r.get::<_,String>(1)?,"admin":r.get::<_,bool>(2)?,"disabled":r.get::<_,bool>(3)?,"created":r.get::<_,u64>(4)?}))).map_err(|e|e.to_string())?.collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string())?;
        Ok(values)
    }
}
fn accounts(app: &AppState) -> Result<Arc<Accounts>, ApiError> {
    app.accounts
        .clone()
        .ok_or_else(|| bad("本机模式不需要账号"))
}

pub(crate) async fn access(
    State(root): State<AppState>,
    mut request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path().to_string();
    let mut tenant = root.clone();
    if let Some(accounts) = &root.accounts {
        if let Some(origin) = request
            .headers()
            .get(header::ORIGIN)
            .and_then(|v| v.to_str().ok())
        {
            if origin != accounts.origin {
                return ApiError(StatusCode::FORBIDDEN, "不允许跨站请求".into()).into_response();
            }
        }
        let identity = match accounts.authenticate(&request) {
            Ok(v) => v,
            Err(e) => return e.into_response(),
        };
        let public = matches!(
            path.as_str(),
            "/api/config" | "/api/auth/login" | "/api/auth/register" | "/api/auth/me"
        );
        if path.starts_with("/api/") && !public {
            let Some(user) = identity.as_ref() else {
                return ApiError(StatusCode::UNAUTHORIZED, "请先登录".into()).into_response();
            };
            if !matches!(
                *request.method(),
                axum::http::Method::GET | axum::http::Method::HEAD
            ) && request
                .headers()
                .get("x-csrf-token")
                .and_then(|v| v.to_str().ok())
                != Some(user.csrf.as_str())
            {
                return ApiError(StatusCode::FORBIDDEN, "登录状态已更新，请刷新页面".into())
                    .into_response();
            }
            if path.starts_with("/api/admin/") && user.user["admin"] != true {
                return ApiError(StatusCode::FORBIDDEN, "需要管理员权限".into()).into_response();
            }
        }
        if let Some(user) = identity {
            match root.for_user(user.user["id"].as_str().unwrap()) {
                Ok(app) => tenant = app,
                Err(e) => return bad(e).into_response(),
            }
            request.extensions_mut().insert(user);
        }
        if matches!(path.as_str(), "/api/auth/login" | "/api/auth/register") {
            let peer = request
                .extensions()
                .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
                .map(|v| v.0.ip().to_string())
                .unwrap_or_else(|| "local".into());
            if let Err(e) = accounts.rate(format!("peer:{peer}"), 60) {
                return e.into_response();
            }
        }
    } else {
        let authority = request
            .headers()
            .get(header::HOST)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let valid = authority
            .parse::<axum::http::uri::Authority>()
            .is_ok_and(|v| matches!(v.host(), "127.0.0.1" | "localhost" | "[::1]"));
        if !valid
            || request
                .headers()
                .get(header::ORIGIN)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v != format!("http://{authority}"))
        {
            return ApiError(StatusCode::FORBIDDEN, "不允许跨站访问本机项目".into())
                .into_response();
        }
    }
    request.extensions_mut().insert(tenant);
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert(header::X_CONTENT_TYPE_OPTIONS, "nosniff".parse().unwrap());
    response
        .headers_mut()
        .insert(header::REFERRER_POLICY, "same-origin".parse().unwrap());
    if path.starts_with("/api/") {
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    }
    response
}
pub(crate) async fn register(
    State(app): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let accounts = accounts(&app)?;
    if !accounts.registration() {
        return Err(ApiError(StatusCode::FORBIDDEN, "管理员已关闭注册".into()));
    }
    let name = username(&body)?.to_string();
    let password = body["password"]
        .as_str()
        .ok_or_else(|| bad("请输入密码"))?
        .to_string();
    let clone = accounts.clone();
    let user = tokio::task::spawn_blocking(move || clone.add_user(&name, &password, false))
        .await
        .map_err(bad)?
        .map_err(bad)?;
    accounts.login_response(user)
}
pub(crate) async fn login(
    State(app): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let accounts = accounts(&app)?;
    let name = username(&body)?.to_string();
    accounts.rate(format!("user:{}", name.to_lowercase()), 20)?;
    let password = body["password"].as_str().unwrap_or("").to_string();
    let clone = accounts.clone();
    let user = tokio::task::spawn_blocking(move || -> Result<Value, String> {
        let row = clone
            .db
            .lock()
            .map_err(|_| "账号服务忙")?
            .query_row(
                "SELECT id,username,password,admin FROM users WHERE username=? AND disabled=0",
                [name],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, bool>(3)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| e.to_string())?;
        if let Some((id, name, hash, admin)) = row {
            if check_password(&password, &hash) {
                return Ok(json!({"id":id,"username":name,"admin":admin,"guest":false}));
            }
        }
        Err("用户名或密码不正确".into())
    })
    .await
    .map_err(bad)?
    .map_err(|e| ApiError(StatusCode::UNAUTHORIZED, e))?;
    accounts.login_response(user)
}
pub(crate) async fn me(identity: Option<Extension<Identity>>) -> Json<Value> {
    Json(
        identity
            .map(|Extension(s)| json!({"user":s.user,"csrf":s.csrf,"expires":s.expires}))
            .unwrap_or(json!({"user":null,"csrf":null})),
    )
}
pub(crate) async fn logout(
    State(app): State<AppState>,
    Extension(user): Extension<Identity>,
) -> Result<Response, ApiError> {
    accounts(&app)?
        .db
        .lock()
        .map_err(|_| bad("账号服务忙"))?
        .execute("DELETE FROM sessions WHERE token=?", [user.token])
        .map_err(bad)?;
    Ok((
        [(
            header::SET_COOKIE,
            "guitarocr-auth=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0",
        )],
        Json(json!({"ok":true})),
    )
        .into_response())
}
pub(crate) async fn password(
    State(app): State<AppState>,
    Extension(user): Extension<Identity>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let accounts = accounts(&app)?;
    accounts.rate(format!("password:{}", user.user["id"]), 10)?;
    let current = body["current"].as_str().unwrap_or("").to_string();
    let new = body["password"].as_str().unwrap_or("").to_string();
    let id = user.user["id"].as_str().unwrap().to_string();
    let clone = accounts.clone();
    tokio::task::spawn_blocking(move || -> Result<(), String> {
        let old = clone
            .db
            .lock()
            .map_err(|_| "账号服务忙")?
            .query_row("SELECT password FROM users WHERE id=?", [&id], |r| {
                r.get::<_, String>(0)
            })
            .map_err(|e| e.to_string())?;
        if !check_password(&current, &old) {
            return Err("当前密码不正确".into());
        }
        let hash = password_hash(&new)?;
        let mut db = clone.db.lock().map_err(|_| "账号服务忙")?;
        let tx = db.transaction().map_err(|e| e.to_string())?;
        tx.execute("UPDATE users SET password=? WHERE id=?", params![hash, id])
            .map_err(|e| e.to_string())?;
        tx.execute("DELETE FROM sessions WHERE user_id=?", [id])
            .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())
    })
    .await
    .map_err(bad)?
    .map_err(bad)?;
    accounts.login_response(user.user)
}
pub(crate) async fn users(State(app): State<AppState>) -> ApiResult {
    let mut rows = accounts(&app)?.users().map_err(bad)?;
    for row in &mut rows {
        let usage = app
            .for_user(row["id"].as_str().unwrap())
            .map_err(bad)?
            .usage();
        row["projects"] = usage["projects"].clone();
        row["gpu_seconds"] = usage["engines"][0]["seconds"].clone();
    }
    Ok(Json(json!(rows)))
}
pub(crate) async fn update_user(
    State(app): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> ApiResult {
    let accounts = accounts(&app)?;
    let clone = accounts.clone();
    let uid = id.clone();
    tokio::task::spawn_blocking(move || -> Result<(), String> {
        let hash = body
            .get("password")
            .and_then(Value::as_str)
            .map(password_hash)
            .transpose()?;
        let mut db = clone.db.lock().map_err(|_| "账号服务忙")?;
        let admin = db
            .query_row("SELECT admin FROM users WHERE id=?", [&uid], |r| {
                r.get::<_, bool>(0)
            })
            .map_err(|_| "用户不存在")?;
        if admin {
            return Err("管理员账号请在设置中修改密码".into());
        }
        let tx = db.transaction().map_err(|e| e.to_string())?;
        if let Some(disabled) = body["disabled"].as_bool() {
            tx.execute(
                "UPDATE users SET disabled=? WHERE id=?",
                params![disabled, uid],
            )
            .map_err(|e| e.to_string())?;
        }
        if let Some(hash) = hash {
            tx.execute("UPDATE users SET password=? WHERE id=?", params![hash, uid])
                .map_err(|e| e.to_string())?;
        }
        tx.execute("DELETE FROM sessions WHERE user_id=?", [uid])
            .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())
    })
    .await
    .map_err(bad)?
    .map_err(bad)?;
    // Any in-flight job keeps its stored results, and can be restarted after account recovery.
    let tenant = app.for_user(&id).map_err(bad)?;
    for (sid, _) in tenant.tasks.snapshots() {
        tenant.tasks.cancel(&sid).map_err(bad)?;
    }
    Ok(Json(json!({"ok":true})))
}
pub(crate) async fn registration(
    State(app): State<AppState>,
    Query(query): Query<HashMap<String, String>>,
) -> ApiResult {
    let enabled = match query.get("enabled").map(String::as_str) {
        Some("true") => true,
        Some("false") => false,
        _ => return Err(bad("注册开关无效")),
    };
    accounts(&app)?.db.lock().map_err(|_|bad("账号服务忙"))?.execute("INSERT INTO settings VALUES('registration',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[enabled.to_string()]).map_err(bad)?;
    Ok(Json(json!({"enabled":enabled})))
}
pub(crate) async fn status(State(app): State<AppState>) -> ApiResult {
    let accounts = accounts(&app)?;
    let mut jobs = vec![];
    for user in accounts.users().map_err(bad)? {
        for (sid, mut job) in app
            .for_user(user["id"].as_str().unwrap())
            .map_err(bad)?
            .tasks
            .snapshots()
        {
            if matches!(job["status"].as_str(), Some("queued" | "running")) {
                job["id"] = json!(format!("{}:{sid}", user["id"].as_str().unwrap()));
                job["username"] = user["username"].clone();
                jobs.push(job);
            }
        }
    }
    Ok(Json(
        json!({"registration":accounts.registration(),"jobs":jobs,"engine":app.pipeline.stats(),"version":env!("CARGO_PKG_VERSION")}),
    ))
}
pub(crate) async fn cancel_job(State(app): State<AppState>, Path(id): Path<String>) -> ApiResult {
    let (user, sid) = id.split_once(':').ok_or_else(|| bad("任务编号无效"))?;
    app.for_user(user)
        .map_err(bad)?
        .tasks
        .cancel(sid)
        .map_err(bad)?;
    Ok(Json(json!({"ok":true})))
}
