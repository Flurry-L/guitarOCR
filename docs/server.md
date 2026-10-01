# 自托管服务

服务端使用与桌面相同的 `guitarocr-backend` 和 `engine`，不需要 Python。服务器模式增加账号、权限及按用户隔离的项目目录；模型实例和下载缓存共享。安装包内的原生后端也可以用服务器参数启动，桌面窗口不必保持运行。

## 安装和运行

下载并解压对应平台的 `GuitarOCR-0.1.0-server-*` 附件，或按[原生打包](native-packaging.md)构建。解压目录包含 `native/`、`ui/` 和 `licenses/`。将其放到 `/opt/guitarocr`，准备可写的数据目录 `/var/lib/guitarocr`，以运行服务的账号创建管理员：

```bash
cd /opt/guitarocr
read -rsp '管理员密码: ' GUITAROCR_ADMIN_PASSWORD
export GUITAROCR_ADMIN_PASSWORD
./native/guitarocr-backend --projects /var/lib/guitarocr/projects --create-admin admin
unset GUITAROCR_ADMIN_PASSWORD
./native/guitarocr-backend --server --bind 127.0.0.1 --port 8080 \
  --public-url https://ocr.example.com \
  --projects /var/lib/guitarocr/projects --models /var/lib/guitarocr/models \
  --resources native --assets ui/workbench --slots 4
```

`--public-url` 是访问服务的公开源地址，不含结尾斜线或路径。密码至少 10 个字符。用反向代理提供 HTTPS，参考 [Caddyfile](../deploy/Caddyfile) 与 [systemd 配置](../deploy/guitarocr.service)；将示例域名和路径换成实际值。

首次识别通过工作台确认并下载模型。已有完整缓存时直接使用。`--slots 1..4` 控制模型并发，CPU 或内存较小的设备建议设为 1。每个账号最多同时提交 4 个任务，整个进程最多 32 个；模型另有有界排队，避免无上限生成线程。

## 页面与账号

首页提供登录和服务入口，`/workbench` 使用共用编辑器。登录后访问自己的项目，管理员可启停注册、启停账号、重设密码及取消任务。注册默认关闭。项目及导出接口均检查登录和归属，写操作检查来源与 CSRF；校对保存使用 revision 防止覆盖另一窗口的修改。

`--api-only` 可以关闭静态网页，只提供 API；由外部网页提供入口时应保持同源。不开启 `--server` 时仅监听本机回环地址，无需账号，适合个人电脑。

普通文件上传最大 200 MB，项目 ZIP 最大 250 MB，每个项目最多 100 页。账号默认限制 200 个项目和 10 GiB 已用空间。删除不需要的项目可以释放空间。

## 保存、恢复与更新

关闭浏览器不会停止任务。用户取消任务时保留已经完成的阶段和小节；服务器重启后，未完成任务标为中断，可在工作台继续。取消一个任务不会关闭其他用户共用的模型进程。

数据库位于项目根目录 `accounts.sqlite3`，账号项目位于 `users/<账号编号>/`。备份整个项目根目录；在线备份数据库使用 SQLite backup，或停止服务后复制目录。模型缓存可独立保存与迁移。

更新时停止服务、替换程序和原生组件，然后用相同数据与模型路径启动。管理员页面链接到发布页，部署程序不从网页拉取并执行仓库源码。任务用时包括预处理与等待，仅用于了解使用情况，不代表 GPU 核心时间。
