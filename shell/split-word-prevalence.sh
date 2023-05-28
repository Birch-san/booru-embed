#!/usr/bin/env bash
# NR == 1 {
#   print $4, $5, $6, $7, $8;
# }
awk -F'\t' 'NR > 1 {
  split($4, meta, " ");
  split($5, general, " ");
  split($6, artist, " ");
  split($7, copyright, " ");
  split($8, character, " ");
  for (label_key in meta)
    counts[meta[label_key]]++;
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
  for (label_key in artist)
    counts[artist[label_key]]++;
  for (label_key in copyright)
    counts[copyright[label_key]]++;
  for (label_key in character)
    counts[character[label_key]]++;
}
END {
  for (key in counts) {
    print counts[key], key;
  }
}' <(head -n 10 <(csv2tsv -H ~/machine-learning/danbooru-bigquery/bq-results-20230521-125535-1684673819225.csv)) | sort -n
# <(csv2tsv -H *.csv) | sort -rn
# <(head -n 10 <(csv2tsv -H *.csv)) | sort -rn