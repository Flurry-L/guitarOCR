export const note = (fret, effects = [], string = 1) => ({ string, fret, effects });
export const beat = (start, notes, effects = [], value = 4) => ({
  start, duration: { value }, status: "normal", notes, effects,
});
export const measure = events => ({ voices: [{ voice: 0, events }] });
export const record = (parsed, bar = 0, fields = {}) => ({
  part_id: "guitar", staff_id: "tab", bar_index: bar, mode: "tab", instrument: "guitar",
  tuning: [64, 59, 55, 50, 45, 40], capo: 2, parsed, ...fields,
});
export const state = (...measures) => ({ metadata: {}, measures });
