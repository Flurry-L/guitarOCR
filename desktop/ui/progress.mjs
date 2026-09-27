// uv reports package starts and completions, but no reliable byte total over
// pipes. Keep those downloads indeterminate; only measured transfers get a %.
export class InstallationProgress {
  constructor(document) {
    this.get = id => document.getElementById(id);
    this.document = document;
  }
  begin(mode) {
    this.steps = mode === 'gpu'
      ? [['python', 'Python'], ['models', '包内模型'], ['ocr', '识别依赖'], ['layout', '版面依赖'], ['check', '启动检查']]
      : [['python', 'Python'], ['editor', '校对依赖'], ['check', '启动检查']];
    this.get('installProgress').hidden = false;
    this.get('installStages').replaceChildren(...this.steps.map(([, label]) => {
      const item = this.document.createElement('li'); item.textContent = label; return item;
    }));
    this.stage = null;
    this.update({stage: 'python', label: '正在准备 Python 3.11', detail: '首次运行需要下载 Python，完成后继续准备所选环境。'});
    this.get('installProgress').scrollIntoView({block: 'center', behavior: 'smooth'});
  }
  update({stage, label, detail = '', completed = null, total = null}) {
    const index = this.steps.findIndex(([key]) => key === stage);
    if (index < 0) return;
    if (this.stage !== stage) {
      this.stage = stage;
      this.downloads = new Map(); this.finished = new Set();
      this.get('downloadList').replaceChildren();
    }
    [...this.get('installStages').children].forEach((item, i) => {
      item.className = i < index ? 'done' : i === index ? 'active' : '';
      if (i === index) item.setAttribute('aria-current', 'step');
      else item.removeAttribute('aria-current');
    });
    this.get('progressLabel').textContent = `${index + 1}/${this.steps.length} ${label}`;
    const bar = this.get('progressBar');
    if (total > 0 && completed !== null) {
      bar.max = total; bar.value = completed;
      const percent = Math.floor(completed / total * 100);
      const amount = total > 1 ? `${(completed / 1024 ** 2).toFixed(0)} / ${(total / 1024 ** 2).toFixed(0)} MiB` : '';
      this.get('progressDetail').textContent = [detail, amount, `${percent}%`].filter(Boolean).join('，');
    } else {
      bar.removeAttribute('value');
      this.get('progressDetail').textContent = detail || '正在处理，耗时取决于网络和磁盘速度。';
    }
  }
  log(line) {
    if (!this.stage) return;
    line = line.replace(/\x1b\[[0-9;]*m/g, '').trim();
    const started = /^Downloading (\S+)(?: \((.+)\))?$/.exec(line);
    const ended = /^Downloaded (\S+)/.exec(line);
    if (started) this.downloads.set(started[1], started[2] || '');
    if (ended) { this.downloads.delete(ended[1]); this.finished.add(ended[1]); }
    if (started || ended) {
      this.get('progressBar').removeAttribute('value');
      this.get('progressDetail').textContent = this.downloads.size
        ? `正在下载 ${this.downloads.size} 个组件，已完成 ${this.finished.size} 个。`
        : `已下载 ${this.finished.size} 个组件，正在继续安装。`;
      this.get('downloadList').replaceChildren(...[...this.downloads].map(([name, size]) => {
        const item = this.document.createElement('li');
        item.textContent = `${name}${size ? `（${size}）` : ''}`; return item;
      }));
    } else if (/^(Prepared|Installed) \d+ packages?/.test(line)) {
      this.get('progressDetail').textContent = line.startsWith('Prepared') ? '下载完成，正在安装组件。' : '组件已安装，正在继续检查。';
      this.get('downloadList').replaceChildren();
    } else if (line.includes('正在使用默认配置和官方源重试')) {
      this.get('progressDetail').textContent = '下载源未能完成安装，正在尝试备用源。';
    }
  }
  end(success) {
    this.stage = null;
    const bar = this.get('progressBar'); bar.max = 1; bar.value = success ? 1 : 0;
    this.get('progressLabel').textContent = success ? '工作台已就绪' : '准备已停止';
    this.get('progressDetail').textContent = success ? '下次启动会复用已安装的环境。' : '已完成的下载会保留。详情见安装与运行日志。';
    this.get('downloadList').replaceChildren();
    if (success) for (const item of this.get('installStages').children) { item.className = 'done'; item.removeAttribute('aria-current'); }
  }
}
