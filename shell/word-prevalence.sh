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
    label_counts[meta[label_key]]++;
  for (label_key in general)
    label_counts[general[label_key]]++;
  for (label_key in artist)
    label_counts[artist[label_key]]++;
  for (label_key in copyright)
    label_counts[copyright[label_key]]++;
  for (label_key in character)
    label_counts[character[label_key]]++;
}
END {
  for (key in label_counts) {
    print label_counts[key], key;
  }
}' ~/machine-learning/danbooru-bigquery-2023-08/danbooru-captions.tsv | sort -rn
# <(head -n 10 ~/machine-learning/danbooru-bigquery-2023-08/danbooru-captions.tsv | sort -rn
# <(head -n 10 <(csv2tsv -H *.csv)) | sort -rn