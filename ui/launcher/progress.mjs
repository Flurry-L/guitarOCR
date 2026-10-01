// Startup only; model download progress belongs to workbench tasks.
export class StartupProgress {
  constructor(document) {
    this.get = id => document.getElementById(id);
    this.active = false;
  }
  begin() {
    this.active = true;
    this.get('startupProgress').hidden = false;
    this.get('progressLabel').textContent = '正在启动原生工作台';
    this.get('progressDetail').textContent = '直接运行应用组件，不安装 Python 或下载模型。';
    this.get('progressBar').removeAttribute('value');
  }
  end(success) {
    this.active = false;
    const bar = this.get('progressBar');
    bar.max = 1; bar.value = success ? 1 : 0;
    this.get('progressLabel').textContent = success ? '工作台已就绪' : '启动已停止';
    this.get('progressDetail').textContent = success ? '项目和缓存保存在本机。' : '请查看运行日志。原有项目不受影响。';
  }
}
