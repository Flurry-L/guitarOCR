# 乐谱数据与 M 文本

GuitarOCR 在识别、编辑和导出时使用三种表示：

| 表示 | 内容 | 代码入口 |
| --- | --- | --- |
| Observation JSON | 单张小节图中的符号、时值和连线 | [Schema](../scorelib/src/observation/schema.rs)、[解析与组装](../scorelib/src/observation.rs) |
| M 文本 | 可直接编辑的单小节音乐 | [解析、格式化与校验](../scorelib/src/score.rs) |
| ScoreIR | 全谱时间轴、音轨、谱表与小节 | [`score_document`](../scorelib/src/score.rs)、[导出](../scorelib/src/music_exports.rs) |

工作流程是 **Observation → 跨节组装 → M 文本 → ScoreIR → GP5 / MusicXML**。Rust 调用者也可用 [types.rs](../scorelib/src/types.rs) 中的 `Observation` 类型读取观察字段；组装与导出接口使用 `serde_json::Value`。

[项目主页](../README.md) · [算法与评测](algorithm-and-models.md) · [开发指南](development.md)

## Observation：一张小节图中的内容

下面是一份 TAB 小节观察。它记录了第一个四分音符、揉弦、下拨和和弦名，以及伸向下一张裁图的延音弧。

```json
{
  "state": {
    "time": { "status": "observed", "value": "4/4" },
    "key": { "status": "absent", "value": null }
  },
  "voices": [{
    "voice": 0,
    "events": [{
      "start": 0,
      "duration": "q",
      "status": "normal",
      "notes": [{ "string": 1, "fret": 5, "effects": ["vib"] }],
      "effects": ["pick_down", "chord:Am7"]
    }]
  }],
  "relations": [{
    "kind": "tie",
    "start": [0, 0, 0],
    "end": "right",
    "shape": "curve"
  }]
}
```

### 状态、事件与音符

`state` 保存拍号 `time`、调号 `key`、谱号 `clef` 和谱号八度变化 `clef_octave`。每项包含 `status` 和 `value`：

| `status` | 含义 | 组装时的处理 |
| --- | --- | --- |
| `observed` | 本节印出了这个记号 | 使用本节值 |
| `absent` | 本节没有新记号 | 沿用前节状态 |
| `unknown` | 记号被裁切或形状读不清 | 留待上下文或编辑补全 |

`voices` 按声部保存事件，每个事件包含起点 `start`、时值 `duration`、类型 `status`、音符 `notes` 和效果 `effects`。起点以 tick 计，**四分音符 = 960 tick**；时值使用 `q`、`e` 等记号。无法从图中确定的起点、时值或品位用 `null` 表示。

音符字段随谱型变化：

| 谱型 | 字段 | 例子 |
| --- | --- | --- |
| TAB | `string`、`fret` | `{"string": 1, "fret": 5}`；第 1 弦是最高音弦，`"x"` 表示闷音 |
| 五线谱 | `staff_position`、`accidental` | `{"staff_position": 0, "accidental": 1}`；底线为 0，每上移一线或一间加 1 |
| 鼓谱 | 谱位、临时记号、`notehead` | 音头可为 `normal`、`x`、`circle-x` 等；鼓件由谱位与音头共同确定 |
| 五线谱＋TAB | `notes` 与 `notation_notes` | 分别保存 TAB 音符和五线谱音头，组装时按音高对应 |

五线谱临时记号用整数半音偏移表示：`-2`、`-1`、`0`、`1`、`2`。未印临时记号用 `null`，印了但读不清用 `"unknown"`。

### 连线与范围

`relations` 描述音符之间的连接和跨拍奏法：

| 关系 | `kind` | 端点 |
| --- | --- | --- |
| 延音、击勾弦、连滑音、换把滑音 | `tie`、`hammer`、`sl`、`ss` | `[声部, 事件, 音符]` |
| 闷音线、延音线、八度线 | `pm`、`let`、`ottava` | `[声部, 事件]` |

端点中的事件和音符从 0 开始计数。伸出裁图的端点写作 `"left"` 或 `"right"`；双谱中的音符关系还用 `representation: "tab"` 或 `"notation"` 指明所在谱表。组装器沿音轨连接相邻小节的开放端点。

## M 文本：编辑一个小节

在工作台选择「小节文本（高级）」即可编辑 M 文本。下面是一个完整的 4/4 小节：第一拍弹 1 弦 5 品，第二拍休止，后两拍弹两音和弦。

```text
M time=4/4 tempo=120 | V0{@0:q:s1f5(vib)<pick_down,chord:Am7> @960:q:r @1920:h:s2f5,s3f5}
```

基本结构是 `M 小节属性 | V0{事件 事件 …}`。声部用 `||` 分隔，声部内的事件用空格分隔，单个事件写成 `@起点:时值:音符`。

### 小节属性

| 属性 | 含义 | 例子 |
| --- | --- | --- |
| `time` | 拍号 | `time=4/4`、`time=6/8` |
| `tempo` | 每分钟四分音符数 | `tempo=120` |
| `key` | 调号的升降数，可附小调标记 | `key=2`、`key=-1Minor` |
| `section` | 段落名 | `section=Chorus` |
| `bar`、`rep`、`alt` | 小节线、反复次数、反复房子 | 由工作台的结构编辑控件生成 |

小节属性按需填写；例如没有新的速度标记时，可以省略 `tempo`。

### 起点与时值

在 4/4 拍中，四拍起点分别为 `@0`、`@960`、`@1920`、`@2880`；在 6/8 拍中，每个八分拍相隔 480 tick。

| 记号 | 时值 | tick |
| --- | --- | ---: |
| `w` | 全音符 | 3840 |
| `h` | 二分音符 | 1920 |
| `q` | 四分音符 | 960 |
| `e` | 八分音符 | 480 |
| `s` | 十六分音符 | 240 |
| `t` | 三十二分音符 | 120 |
| `f` | 六十四分音符 | 60 |

后缀 `.` 表示附点，`..` 表示双附点；`[p:q]` 表示 p 个音占 q 个同类音的时长。例如 `e[3:2]` 是八分三连音，每音 320 tick。下面将第一拍拆成三个音，其余三拍休止：

```text
M time=4/4 | V0{@0:e[3:2]:p60 @320:e[3:2]:p62 @640:e[3:2]:p64 @960:h.:r}
```

### 音符、休止与多声部

| 写法 | 含义 |
| --- | --- |
| `s1f5`、`s6fx` | 1 弦 5 品、6 弦闷音 |
| `p60`、`s1f5p69` | MIDI 音高 60（C4）；弦品与音高一起保存 |
| `p36`、`p38`、`p42` | 鼓谱中的底鼓、军鼓、闭镲；鼓件使用同一套 `p` 记号 |
| `s1f0,s2f1,s3f2` | 同一事件的三个音，以逗号连接 |
| `r` | 休止 |
| `e` | 空拍 |
| `^` | 复制前一事件的音符与效果，使用当前起点和时值 |

两个声部同时开始的例子：

```text
M time=4/4 | V0{@0:h:p64 @1920:h:p67} || V1{@0:w:p48}
```

### 奏法与事件效果

音符后的圆括号 `(...)` 保存揉弦、延音等奏法；事件末尾的尖括号 `<...>` 保存拨弦方向、力度、和弦名等效果。同一括号内用逗号分隔。

| 位置 | 示例 |
| --- | --- |
| 音符奏法 | `s1f5(vib)`、`p64(tie)`、`s6f0(pm)`、`s1f12(harm:natural)` |
| 推弦 | `s1f7(bend:bend:100)`，每半音记为 50，100 表示全音 |
| 事件效果 | `<pick_down,dyn:90,chord:Am7>`、`<tempo:100>` |
| 文本 | `<text:let%20ring>`，空格与分隔字符使用百分号编码 |

编辑器使用的音高已完成乐器移调与谱号八度换算。吉他、贝斯的 `pitch` 是加变调夹前的音高，播放时再加 `capo`；其他旋律乐器使用实际发声音高，鼓谱使用 MIDI 鼓件编号。

## ScoreIR：组织整首乐谱

`score_document` 输出的格式标识为 `"schema": "guitarocr.score/2"`，时间单位为 `"ticks_per_quarter": 960`。常用字段如下：

| 层级 | 字段 | 用途 |
| --- | --- | --- |
| 整谱 | `title`、`artist`、`subtitle`、`instructions` | 标题、署名与谱头文字 |
| 时间轴 | `timeline[].index`、`time_signature`、`tempo_quarter` | 对齐各音轨的小节与速度 |
| 音轨 | `parts[].id`、`name`、`instrument`、`midi_program` | 标识乐器及 MIDI 音色 |
| 音高上下文 | 音轨的 `tuning`、`capo`、`pitch_reference` | 定弦、变调夹和音高基准 |
| 谱表 | `parts[].staves[].id`、`measures` | 一个乐器的不同书写层 |
| 小节 | `index`、`voices`、`chord_annotations`、`pitch_context` | 音乐事件、和弦注释和谱号／移调信息 |
| 整谱标注 | `annotations` | 和弦指法库等带音轨归属的标注 |

音轨内的小节按 `index` 排序，并与全局 `timeline` 对齐。GP5 和 MusicXML 导出器从这一结构读取音乐与排版信息。
