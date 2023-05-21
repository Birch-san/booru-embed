# Booru-embed

Gonna try and make a transformer for embedding Danbooru labelsets.

Goals:

- efficient tokenizer
  - not splitting Danbooru labels into unigrams
  - not allocating vocabulary for unpopular labels
- long context length
  - analyse how "long" Danbooru captions tend to be; aim to support the most common lengths.
  - the efficient tokenizer will also help to avoid word-splitting, so labels require one token instead of many.
- low-dimensionality embedding
  - fewer dims: easier for another model to condition on
- consider using other metadata as part of the embedding
  - e.g. rating and score
  - could use [score categories](https://gist.github.com/harubaru/8581e780a1cf61352a739f2ec2eef09b?permalink_comment_id=4422511#prompting) like WD1.4 did

Mostly trying to reproduce what marunine's T5 Danbooru transformer was able to do. Sounded from that like t5-small can perform sufficiently well. Training should take about a day on a high-end card. Probably faster, now that flash attention and 4090s exist.

There has been a lot of LLM progress since T5, so maybe there is a better-performing model to reach for.

Danbooru captions present some special challenges/opportunities:

- predicting duplicate tags is never the right answer
  - we could make our loss function punish this. LLaMa had some repetition-avoidance thing: we could check how that works.
- captions are unordered
  - this might have implications for masking. ordinarily a language transformer might use a causal mask, to avoid attending to future information in the sentence. but maybe this isn't a problem for us.

Implementation concerns:

- HF transformers library doesn't use flash attention everywhere. So we should be prepared to modify whatever model we use.
- Don't bother finetuning a pretrained base. We are changing the tokenizer.

## Getting the input data

Use Google BigQuery to export Danbooru labels from `danbooru1.danbooru_public.posts`:  
https://console.cloud.google.com/bigquery

Create a new BigQuery project. You can get a feel for what columns exist, like so:

```sql
SELECT * FROM danbooru1.danbooru_public.posts LIMIT 1
```

We can export results to Google Drive as 1GB csvs.
You can use `LIMIT` and `OFFSET` to request (for example) 2500000 records at a time: each csv will be about ~972MB.  
There are currently ~5267057 Danbooru posts that satisfy this filter (i.e. non-explicit, non-banned):

```sql
SELECT id, rating, score, tag_string_meta, tag_string_general, tag_string_artist, tag_string_copyright, tag_string_character FROM danbooru1.danbooru_public.posts WHERE is_deleted = false and is_banned = false AND rating IN ('g', 's', 'q') and is_flagged = false LIMIT 2500000 OFFSET 0
```

## Analysing the input data

First, let's convert csv to tsv:

```bash
head -n 2 bq-results-20230520-201605-1684613827694.csv | csvformat -T
```

Then we can use awk to count tag occurrences, and print a sorted list.  
We can work out what %ile we want to keep, based on how much vocab size it would cost.

Might want to count each category of tag separately. Because an artist can be prolific and yet have fewer occurrences than an unremarkable general tag.

Once we've decided on a vocab list, we can move to the next step:

## Processing the csv for fast dataloading

Let's have it be just one long binary file. Should be small enough (<2GB) that we don't need to shard it.

We'd encode every line in the csv as a binary recrod, which just holds references to each tag's id in our vocabulary.
If our vocab size is under 131072, then we can use 16-bit vocab references.  

We could reserve some numbers for use as field delimiters or record delimiters. Or we could use sequences of NUL bytes.

Dataloader would mmap the large file. I think child processes can share the file descriptor.

To grab random-ish records from it, you would:

- seek to random locations in the file
  - maybe do an n-megabyte read
- discard bytes until see the record delimiter
- read until you reach the desired number of records
  - if you reach the end of your page or of the file: seek to another random location and continue