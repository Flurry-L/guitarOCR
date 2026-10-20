// Musical names and conversions shared by editing, summaries and engraving.
const majorTonics = ["FFlat", "CFlat", "GFlat", "DFlat", "AFlat", "EFlat", "BFlat", "F", "C", "G", "D", "A", "E", "B", "FSharp", "CSharp", "GSharp"];
const minorTonics = ["DFlat", "AFlat", "EFlat", "BFlat", "F", "C", "G", "D", "A", "E", "B", "FSharp", "CSharp", "GSharp", "DSharp", "ASharp", "ESharp"];
const displayFifths = Array.from({ length: 15 }, (_, i) => i - 7);
const tonic = (fifths, minor) => (minor ? minorTonics : majorTonics)[fifths + 8];
export const keyCode = (fifths, minor) => tonic(fifths, minor) + (minor ? "Minor" : "Major");
export const keyName = (fifths, minor) => tonic(fifths, minor).replace("Flat", "♭").replace("Sharp", "♯");
export const keyChoices = [false, true].flatMap(minor =>
  displayFifths.map(fifths => ({
    value: keyCode(fifths, minor),
    label: `${keyName(fifths, minor)} ${minor ? "小调" : "大调"}`,
  })));

export function writtenKey(key, shift) {
  if (typeof key === "number") return key; // score_state keys are already written.
  const normalized = String(key)
    .replace(/[♯#]/g, "Sharp").replace(/[♭b]/g, "Flat")
    .replace(/(Major|Minor)(Sharp|Flat)/, "$2$1");
  const minor = normalized.includes("Minor");
  const name = normalized.replace(/Major|Minor/g, "");
  const index = (minor ? minorTonics : majorTonics).indexOf(name);
  if (index < 0) throw new Error(`无法识别调号：${key}`);
  const fifths = index - 8;
  if (!shift || shift % 12 === 0) return fifths < -7 ? fifths + 12 : fifths > 7 ? fifths - 12 : fifths;
  const pitchClass = (((7 * fifths - shift) % 12) + 12) % 12;
  return displayFifths
    .filter(k => (((7 * k) % 12) + 12) % 12 === pitchClass)
    .sort((a, b) => Math.abs(a) - Math.abs(b) || Math.abs(a - fifths) - Math.abs(b - fifths))[0];
}

export const durations = [
  ["w", 1, "全音符"], ["h", 2, "二分音符"], ["q", 4, "四分音符"],
  ["e", 8, "八分音符"], ["s", 16, "十六分音符"],
  ["t", 32, "三十二分音符"], ["f", 64, "六十四分音符"],
];
export const durationName = value => durations.find(([, denominator]) => denominator === value)?.[2] || `${value} 分音符`;

export function parseDuration(token) {
  const match = /^(w|h|q|e|s|t|f|d[0-9]+)(\.\.|\.)?(?:\[([0-9]+):([0-9]+)\])?$/.exec(token);
  if (!match) return null;
  const value = durations.find(([symbol]) => symbol === match[1])?.[1] ?? Number(match[1].slice(1));
  const enters = Number(match[3] || 1), times = Number(match[4] || 1);
  return value > 0 && enters > 0 && times > 0
    ? { value, dotted: match[2] === ".", double_dotted: match[2] === "..", tuplet_enters: enters, tuplet_times: times }
    : null;
}

export function describeDuration(token) {
  const duration = parseDuration(token);
  if (!duration) return token;
  const prefix = duration.double_dotted ? "双附点" : duration.dotted ? "附点" : "";
  const tuplet = token.includes("[") ? `（${duration.tuplet_enters}:${duration.tuplet_times} 连音）` : "";
  return prefix + durationName(duration.value) + tuplet;
}

export function decodeScoreText(value) {
  try { return decodeURIComponent(value); } catch { return value; }
}
