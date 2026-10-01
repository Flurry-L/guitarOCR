//! Multipart ingestion and page normalization before any model is loaded.
use crate::bad;
use crate::image_boundary;
use crate::project::write_session_atomic;
use crate::view;
use crate::ApiError;
use crate::ApiResult;
use crate::AppState;
use axum::{extract::Multipart, http::StatusCode, Extension, Json};
use serde_json::json;
use std::{fs, io::Write, path::Path};

pub(crate) async fn upload(
    Extension(app): Extension<AppState>,
    mut multipart: Multipart,
) -> ApiResult {
    app.check_quota()?;
    let staging = tempfile::Builder::new()
        .prefix(".upload-")
        .tempdir_in(&app.projects)
        .map_err(bad)?;
    fs::create_dir(staging.path().join("uploads")).map_err(bad)?;
    let mut names = Vec::new();
    let mut files = Vec::new();
    let mut total = 0usize;
    let mut action = "import".to_string();
    let mut mode = "auto".to_string();
    let mut allow_download = false;
    while let Some(mut field) = multipart.next_field().await.map_err(bad)? {
        match field.name() {
            Some("allow_download") => {
                allow_download = field.text().await.map_err(bad)? == "true";
                continue;
            }
            Some("action") => {
                action = field.text().await.map_err(bad)?;
                continue;
            }
            Some("mode") => {
                mode = field.text().await.map_err(bad)?;
                continue;
            }
            Some("files") => {}
            _ => return Err(bad("无效的上传字段")),
        }
        if files.len() >= 100 {
            return Err(bad("每个项目最多100个文件"));
        }
        let name = field
            .file_name()
            .unwrap_or("score")
            .replace('\\', "/")
            .rsplit('/')
            .next()
            .unwrap_or("score")
            .to_string();
        let extension = Path::new(&name)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        if !matches!(
            extension.as_str(),
            "pdf" | "png" | "jpg" | "jpeg" | "bmp" | "tif" | "tiff"
        ) {
            return Err(bad("支持PDF、PNG、JPEG、BMP与TIFF"));
        }
        let filename = format!("{:03}.{extension}", files.len() + 1);
        let mut file =
            fs::File::create(staging.path().join("uploads").join(&filename)).map_err(bad)?;
        while let Some(chunk) = field.chunk().await.map_err(bad)? {
            total += chunk.len();
            if total > 200 * 1024 * 1024 {
                return Err(ApiError(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "总上传大小不能超过200MB".into(),
                ));
            }
            file.write_all(&chunk).map_err(bad)?;
        }
        files.push(filename);
        names.push(name);
    }
    if files.is_empty() {
        return Err(bad("请选择PDF或图片"));
    }
    if !matches!(mode.as_str(), "auto" | "tab" | "notation" | "both")
        || !matches!(action.as_str(), "import" | "full")
    {
        return Err(bad("无效的处理方式或谱面类型"));
    }
    if action == "full" && !app.pipeline.available() {
        return Err(ApiError(
            StatusCode::CONFLICT,
            "原生识别组件尚未准备，请先选择导入并检查页面".into(),
        ));
    }

    let mut id = [0u8; 16];
    getrandom::fill(&mut id).map_err(bad)?;
    let sid = id.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let directory = view::directory(&app.projects, &sid).map_err(bad)?;
    fs::rename(staging.path(), &directory).map_err(bad)?;
    let _ = staging.keep();
    let inputs = files
        .iter()
        .map(|name| directory.join("uploads").join(name))
        .collect::<Vec<_>>();
    let root = app.projects.clone();
    let id = sid.clone();
    let native = app.native.clone();
    let pipeline = app.pipeline.clone();
    let task_action = action.clone();
    let job=app.tasks.submit(sid.clone(),"正在导入页面",&action,move |progress|{
        let pages_dir=directory.join("pages");fs::create_dir(&pages_dir).map_err(|e|e.to_string())?;
        let mut pages=Vec::new();let mut total_pixels=0u64;
        for (index,input) in inputs.iter().enumerate(){
            progress.checkpoint()?;
            if input.extension().and_then(|s|s.to_str())==Some("pdf") {
                let library=native.pdfium.as_ref().ok_or("原生PDF组件尚未准备，当前可先导入图片或项目ZIP")?;
                let rendered=image_boundary::render_pdf_pages(library,input,&pages_dir.join(format!("pdf-{index}")),100-pages.len())?;
                for page in rendered {
                    total_pixels+=u64::from(page.width)*u64::from(page.height);
                    check_size(page.width,page.height,total_pixels)?;
                    pages.push(json!({"image":page.path,"source_pdf":input,"pdf_page":page.page,"vector":false,"width":page.width,"height":page.height}));
                }
            }else{
                // Image decoder supplies all frames; page count is bounded before
                // materializing them, so a TIFF cannot silently lose later pages.
                image_boundary::for_each_image_page(input,100-pages.len(),|image|{
                    progress.checkpoint()?;
                    total_pixels+=u64::from(image.width())*u64::from(image.height());
                    check_size(image.width(),image.height(),total_pixels)?;
                    let output=pages_dir.join(format!("page-{:03}.png",pages.len()+1));
                    image.save(&output).map_err(|e|e.to_string())?;
                    pages.push(json!({"image":output,"source_pdf":null,"pdf_page":null,"vector":false,"width":image.width(),"height":image.height()}));                    Ok(())
                })?;
            }
            if pages.len()>100{return Err("每个项目最多100页".into());}
            progress.update(json!({"done":index+1,"total":inputs.len(),"message":"正在导入页面"}))?;
        }
        if pages.is_empty(){return Err("上传内容没有可读取页面".into());}
        progress.checkpoint()?;
        let state=json!({"id":id,"pages":pages,"inputs":inputs,"input_names":names,"mode":mode,"mode_setting":mode,"boxes":[],"layout":null,"info":null,"recognition":null,"export":null,"revision":0});
        write_session_atomic(&root,&id,&state).map_err(|e|e.to_string())?;
        if task_action=="full" { pipeline.process(&root,&id,&json!({"mode":mode,"allow_download":allow_download}),&progress)?; }
        Ok(())
    });
    match job {
        Ok(value) => Ok(Json(value)),
        Err(error) => {
            let _ = fs::remove_dir_all(app.projects.join(&sid));
            Err(bad(error))
        }
    }
}
fn check_size(width: u32, height: u32, total: u64) -> Result<(), String> {
    if u64::from(width) * u64::from(height) > 40_000_000 || total > 200_000_000 {
        Err("页面尺寸过大，请拆分为多个项目".into())
    } else {
        Ok(())
    }
}
