//! Existing GP5 navigation-sign positions and per-measure swing metadata.
use super::*;
pub(super) const SIGNS: [&str; 19] = [
    "Coda",
    "Double Coda",
    "Segno",
    "Segno Segno",
    "Fine",
    "Da Capo",
    "Da Capo al Coda",
    "Da Capo al Double Coda",
    "Da Capo al Fine",
    "Da Segno",
    "Da Segno al Coda",
    "Da Segno al Double Coda",
    "Da Segno al Fine",
    "Da Segno Segno",
    "Da Segno Segno al Coda",
    "Da Segno Segno al Double Coda",
    "Da Segno Segno al Fine",
    "Da Coda",
    "Da Double Coda",
];
pub(super) fn signs(rows: &[Option<&Value>]) -> R<Vec<String>> {
    let mut out = vec![];
    for r in rows.iter().flatten() {
        for field in ["direction", "from_direction"] {
            let sign = txt(r, field, "");
            if sign.is_empty() {
                continue;
            }
            if !SIGNS.contains(&sign) {
                return Err(format!("Unknown score navigation sign {sign:?}"));
            }
            if !out.iter().any(|s| s == sign) {
                out.push(sign.to_owned())
            }
        }
    }
    Ok(out)
}
pub(super) fn feel(rows: &[Option<&Value>]) -> R<u8> {
    let mut found = None;
    for r in rows.iter().flatten() {
        if r["triplet_feel"].is_null() {
            continue;
        }
        let value = match txt(r, "triplet_feel", "none") {
            "none" => 0,
            "eighth" => 1,
            "sixteenth" => 2,
            _ => return Err("Unknown swing/triplet feel".into()),
        };
        if found.is_some_and(|f| f != value) {
            return Err("Conflicting swing between synchronized staves".into());
        }
        found = Some(value);
    }
    Ok(found.unwrap_or(0))
}
