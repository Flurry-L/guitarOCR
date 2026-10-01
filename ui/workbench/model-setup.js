// Shared by local editing and the hosted upload page.
let downloadAllowed = false;
export function modelOptions(config, body) {
  if (config.native_runtime_error) {
    throw new Error(`${config.server ? '服务器' : '本机'}识别组件未能加载：${config.native_runtime_error}。已保存项目仍可校对和导出。`);
  }
  if (config.requires_model_confirmation && !downloadAllowed) {
    const size = ((config.model_download_bytes || 0) / 1e9).toFixed(2);
    const location = config.server ? '服务器' : '这台电脑';
    if (!confirm(`${location}将校验模型缓存，缺失或损坏时最多下载约 ${size} GB。已有完整缓存不会重复下载，建议预留 5 GB 磁盘空间。继续准备？`)) {
      throw new Error('已取消模型准备，当前项目和所选文件保留。');
    }
    downloadAllowed = true;
  }
  return { ...(body || {}), allow_download: downloadAllowed };
}
