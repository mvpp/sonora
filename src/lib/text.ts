// Joining transcript segments. Chinese and Japanese are written without spaces
// between words, so a space only goes after a segment that ends in a
// space-delimited script (Latin, Cyrillic, Hangul…).

// CJK punctuation, kana, Han ideographs and full-width forms (，。！？：；…).
const NO_SPACE_AFTER =
  /[　-〿぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯]$/u;

/** Separator to put after `segment` before whatever is dictated next. */
export function segmentGap(segment: string): string {
  return NO_SPACE_AFTER.test(segment) ? "" : " ";
}

/** Join transcript segments with script-appropriate separators. */
export function joinSegments(segments: string[]): string {
  return segments
    .map((s) => s.trim())
    .filter(Boolean)
    .reduce((acc, s) => (acc ? acc + segmentGap(acc) + s : s), "");
}
