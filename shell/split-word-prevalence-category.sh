#!/usr/bin/env bash
DIRNAME="$(dirname "$0")"
OUT_DIR="$DIRNAME/../out_prevalence_category"

mkdir -p "$OUT_DIR"

awk -F'\t' 'NR > 1 {
  split($4, meta, " ");
  for (label_key in meta)
    counts[meta[label_key]]++;
}
END {
  for (key in counts) {
    print counts[key], key;
  }
}' ~/machine-learning/danbooru-bigquery-2023-08/danbooru-captions.tsv | sort -rn > "$OUT_DIR/meta.txt"

awk -F'\t' 'NR > 1 {
  split($6, artist, " ");
  for (label_key in artist)
    counts[artist[label_key]]++;
}
END {
  for (key in counts) {
    print counts[key], key;
  }
}' ~/machine-learning/danbooru-bigquery-2023-08/danbooru-captions.tsv | sort -rn > "$OUT_DIR/artist.txt"

awk -F'\t' 'NR > 1 {
  split($7, copyright, " ");
  for (label_key in copyright)
    counts[copyright[label_key]]++;
}
END {
  for (key in counts) {
    print counts[key], key;
  }
}' ~/machine-learning/danbooru-bigquery-2023-08/danbooru-captions.tsv | sort -rn > "$OUT_DIR/copyright.txt"

awk -F'\t' 'NR > 1 {
  split($8, character, " ");
  for (label_key in character)
    counts[character[label_key]]++;
}
END {
  for (key in counts) {
    print counts[key], key;
  }
}' ~/machine-learning/danbooru-bigquery-2023-08/danbooru-captions.tsv | sort -rn > "$OUT_DIR/character.txt"

awk -F'\t' 'NR > 1 {
  split($5, general, " ");
  for (label_key in general) {
    label = general[label_key];
    if (length(label) <= 4) {
      counts[label]++;
    } else {
      rest = label;
      continue_label_loop = 0;
      while (has_qualifier = match(rest, /^(.*)_\(([^)]*?)\)$/, qualifier_match)) {
        rest = qualifier_match[1];
        qualifier = qualifier_match[2];
        counts[qualifier]++;
        if (qualifier == "cosplay") {
          # _(cosplay) qualifies a name label, so what precedes it can be used
          # without any splitting
          counts[rest]++;
          continue_label_loop = 1;
          break;
        }
      }
      if (continue_label_loop)
        continue;
      split(rest, parts, /[-_]/);
      for(i in parts) {
        part = parts[i];
        if (part) {
          counts[part]++;
        }
      }
    }
  }
}
END {
  for (key in counts) {
    print counts[key], key;
  }
}' ~/machine-learning/danbooru-bigquery-2023-08/danbooru-captions.tsv | sort -rn > "$OUT_DIR/general.txt"