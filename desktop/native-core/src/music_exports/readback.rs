//! Bounded native reader for the exact GP5.10 subset emitted by this crate.
//! This is a verification gate, not a general-purpose GP importer.
use super::*;
struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> R<&'a [u8]> {
        let end = self.at.checked_add(n).ok_or("GP5 offset overflow")?;
        let v = self.data.get(self.at..end).ok_or("Truncated GP5 output")?;
        self.at = end;
        Ok(v)
    }
    fn u(&mut self) -> R<u8> {
        Ok(self.take(1)?[0])
    }
    fn i(&mut self) -> R<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn count(&mut self) -> R<usize> {
        let n = self.i()?;
        if !(0..=1_000_000).contains(&n) {
            return Err("Invalid GP5 length".into());
        }
        Ok(n as usize)
    }
    fn bend(&mut self) -> R<()> {
        let kind = self.u()?;
        if kind > 11 {
            return Err("Invalid GP5 bend type".into());
        }
        self.i()?;
        let count = self.count()?;
        if count > 4096 {
            return Err("Too many GP5 bend points".into());
        }
        for _ in 0..count {
            let x = self.i()?;
            if !(0..=60).contains(&x) {
                return Err("Invalid GP5 bend position".into());
            }
            self.i()?;
            if self.u()? > 1 {
                return Err("Invalid bend vibrato flag".into());
            }
        }
        Ok(())
    }
    fn string(&mut self) -> R<()> {
        let size = self.count()?;
        let len = self.u()? as usize;
        if size != len + 1 {
            return Err("GP5 string length mismatch".into());
        }
        self.take(len)?;
        Ok(())
    }
}
pub(super) fn verify(bytes: &[u8], expected: &Value) -> R<()> {
    let mut r = Reader { data: bytes, at: 0 };
    r.take(31)?;
    for _ in 0..9 {
        r.string()?
    }
    if r.i()? != 0 {
        return Err("Unexpected GP5 notice".into());
    }
    r.take(4)?;
    for _ in 0..5 {
        r.take(4)?;
        let n = r.count()?;
        r.take(n)?;
    }
    r.take(19 + 30)?;
    for _ in 0..10 {
        r.string()?
    }
    r.string()?;
    r.take(10 + 768 + 38 + 4)?;
    let measures = r.count()?;
    let tracks = r.count()?;
    for bi in 0..measures {
        if bi > 0 {
            r.take(1)?;
        }
        let flags = r.u()?;
        if flags & 1 != 0 {
            r.take(1)?;
        }
        if flags & 2 != 0 {
            r.take(1)?;
        }
        if flags & 8 != 0 {
            r.take(1)?;
        }
        if flags & 32 != 0 {
            r.string()?;
            r.take(4)?;
        }
        if flags & 64 != 0 {
            r.take(2)?;
        }
        if flags & 16 != 0 {
            r.take(1)?;
        }
        if flags & 3 != 0 {
            r.take(4)?;
        }
        if flags & 16 == 0 {
            r.take(1)?;
        }
        r.take(1)?;
    }
    for ti in 0..tracks {
        if ti == 0 {
            r.take(1)?;
        }
        r.take(1 + 41)?;
        let n = r.i()?;
        if !(1..=7).contains(&n) {
            return Err("Invalid GP5 output string count".into());
        }
        r.take(28 + 4 + 8 + 4 + 4 + 4 + 2 + 3 + 8 + 4 + 12 + 16 + 4)?;
        r.string()?;
        r.string()?;
    }
    r.take(1)?;
    let mut result = vec![];
    let mut previous = vec![vec![HashMap::new(); 2]; tracks];
    for mi in 0..measures {
        for (ti, voices) in previous.iter_mut().enumerate() {
            for (vi, prev) in voices.iter_mut().enumerate() {
                let mut cursor = 0i64;
                for _ in 0..r.count()? {
                    let flags = r.u()?;
                    let status = if flags & 64 != 0 { r.u()? } else { 1 };
                    let power = r.u()? as i8 as i32 + 2;
                    if !(0..=7).contains(&power) {
                        return Err("Invalid GP5 duration byte".into());
                    }
                    let mut duration = 3840i64 / (1i64 << power);
                    if flags & 1 != 0 {
                        duration = duration * 3 / 2
                    }
                    let mut exact_numerator = duration as i128;
                    let mut exact_denominator = 1;
                    if flags & 32 != 0 {
                        let enters = r.i()?;
                        let times = match enters {
                            3 => 2,
                            5..=7 => 4,
                            9..=13 => 8,
                            _ => return Err("Invalid GP5 tuplet".into()),
                        };
                        exact_numerator *= times as i128;
                        exact_denominator = enters as i128;
                        duration = duration * times / enters as i64;
                    }
                    if flags & 2 != 0 {
                        match r.u()? {
                            0 => {
                                r.string()?;
                                if r.i()? != 0 {
                                    return Err("Unexpected old-style GP5 frame".into());
                                }
                            }
                            1 => {
                                let chord = r.take(106)?;
                                if chord[74] > 5 {
                                    return Err("Invalid GP5 barre count".into());
                                }
                            }
                            _ => return Err("Invalid GP5 chord schema".into()),
                        }
                    }
                    if flags & 4 != 0 {
                        r.string()?
                    }
                    if flags & 8 != 0 {
                        let f1 = r.u()?;
                        let f2 = r.u()?;
                        if f1 & !0x72 != 0 || f2 & !7 != 0 {
                            return Err("Invalid GP5 beat technique flags".into());
                        }
                        if f1 & 32 != 0 {
                            r.take(1)?;
                        }
                        if f2 & 4 != 0 {
                            r.bend()?;
                        }
                        if f1 & 64 != 0 {
                            r.take(2)?;
                        }
                        if f2 & 2 != 0 {
                            r.take(1)?;
                        }
                    }
                    if flags & 16 != 0 {
                        r.take(1 + 16 + 6)?;
                        r.string()?;
                        let bpm = r.i()?;
                        if !(1..=1000).contains(&bpm) {
                            return Err("Invalid GP5 tempo".into());
                        }
                        r.take(4)?;
                        r.string()?;
                        r.string()?;
                    }
                    let strings = r.u()?;
                    let mut notes = vec![];
                    for string in 1..=7 {
                        if strings & (1 << (7 - string)) == 0 {
                            continue;
                        }
                        let nf = r.u()?;
                        let typ = r.u()?;
                        if nf & 16 != 0 {
                            r.take(1)?;
                        }
                        let encoded = r.u()? as i8 as i64;
                        let fret = if typ == 2 {
                            *prev
                                .get(&string)
                                .ok_or("GP5 tie has no native readback origin")?
                        } else {
                            encoded
                        };
                        r.take(1)?;
                        if nf & 8 != 0 {
                            let f1 = r.u()?;
                            let f2 = r.u()?;
                            if f1 & !0x1b != 0 || f2 & !0x7f != 0 {
                                return Err("Invalid GP5 note technique flags".into());
                            }
                            if f1 & 1 != 0 {
                                r.bend()?;
                            }
                            if f1 & 16 != 0 {
                                r.take(5)?;
                            }
                            if f2 & 4 != 0 {
                                r.take(1)?;
                            }
                            if f2 & 8 != 0 {
                                r.take(1)?;
                            }
                            if f2 & 16 != 0 {
                                match r.u()? {
                                    1 | 4 | 5 => {}
                                    2 => {
                                        r.take(3)?;
                                    }
                                    3 => {
                                        r.take(1)?;
                                    }
                                    _ => return Err("Invalid GP5 harmonic tag".into()),
                                }
                            }
                            if f2 & 32 != 0 {
                                r.take(2)?;
                            }
                        }
                        prev.insert(string, fret);
                        notes.push(json!([string, fret, typ]));
                    }
                    r.take(2)?;
                    let exact_duration =
                        crate::score::Rational::new(exact_numerator, exact_denominator)?;
                    result.push(json!({"measure":mi,"track":ti,"voice":vi,"start":cursor,"duration":duration,"duration_fraction":[exact_duration.numerator.to_string(),exact_duration.denominator.to_string()],"status":status,"notes":notes}));
                    if status != 0 {
                        cursor += duration;
                    }
                }
            }
            r.take(1)?;
        }
    }
    if r.at != bytes.len() {
        return Err("GP5 output contains trailing data".into());
    }
    if json!(result) != *expected {
        return Err(
            "Native GP5 readback changed notes, ties, durations or voice timing; no output written"
                .into(),
        );
    }
    Ok(())
}
