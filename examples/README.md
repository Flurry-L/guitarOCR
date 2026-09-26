# 示例乐谱

`demo.pdf` 和 `demo.png` 是项目自编的四小节短谱 **First Steps**；`expected.score.txt` 与 `expected.gp5` 是其预期转录。它们用于安装验收、熟悉编辑器和格式示范，不构成识别准确率评测集。

在工作台上传 PDF 或 PNG，检测并检查四个小节框，再识别、校对并导出。可以直接打开 `expected.gp5` 对照音符。工作台不会自动把示例作为你的当前项目。

重新生成：

```bash
uv run --no-sync python -m examples.generate
```

只验证文本到 GP5 的转换：

```bash
uv run --no-sync python -m gp5_export.writer examples/expected.score.txt output/example.gp5 --mode tab
```
