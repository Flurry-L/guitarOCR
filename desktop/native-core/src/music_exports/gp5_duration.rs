//! Legacy GP5 projection contracts. M2 stays exact; GP5's existing codec keeps
//! only its single-dot flag; supported tuplets retain rational durations.
use super::*;
pub(super) fn rest_events(mut total: i64) -> R<Vec<Value>> {
    if total < 0 {
        return Err("Negative GP5 silence span".into());
    }
    let mut out = vec![];
    // Matches shared.m2.full_measure_rest_target, including its 10-tick guard.
    let durations = [
        (3840, 1, false, 1, 1),
        (2880, 2, true, 1, 1),
        (1920, 2, false, 1, 1),
        (1440, 4, true, 1, 1),
        (960, 4, false, 1, 1),
        (720, 8, true, 1, 1),
        (480, 8, false, 1, 1),
        (360, 16, true, 1, 1),
        (240, 16, false, 1, 1),
        (180, 32, true, 1, 1),
        (120, 32, false, 1, 1),
        (60, 64, false, 1, 1),
        (40, 64, false, 3, 2),
        (30, 128, false, 1, 1),
        (20, 128, false, 3, 2),
    ];
    while total > 0 {
        let Some(&(ticks, value, dotted, enters, times)) = durations
            .iter()
            .find(|(n, _, _, _, _)| *n <= total && total - n != 10)
        else {
            return Err("GP5 silence gap has no exact legacy rest projection".into());
        };
        out.push(json!({"duration":{"value":value,"dotted":dotted,"double_dotted":false,"tuplet_enters":enters,"tuplet_times":times},"notes":[],"status":"rest"}));
        total -= ticks;
    }
    Ok(out)
}
