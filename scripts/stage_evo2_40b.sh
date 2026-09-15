#!/bin/bash
# Stage Evo 2 40B from Mila's read-only weight store onto a compute cluster.
#
# The checkpoint ships as two shards because of HuggingFace file-size limits.
# evo2.models.load_evo2_model concatenates them before calling load_checkpoint,
# so staging is not finished at transfer: the merge is required, and it is a
# plain byte concatenation in part order.
#
# Shards are kept rather than removed (the package deletes them) -- re-merging
# costs minutes, re-transferring costs 82 GB.
set -uo pipefail

PART_BYTES=41126745847
MERGED_BYTES=82253491694
SRC="${EVO2_40B_SRC:-mila:/network/weights/evo2/evo2_40b}"
DEST="${EVO2_40B_DEST:-${SCRATCH:-/scratch}/merging-dogma/models/evo2_40b}"

verify() {
  local dir="$1" bad=0 size
  for part in evo2_40b.pt.part0 evo2_40b.pt.part1; do
    if [[ ! -f "$dir/$part" ]]; then echo "missing $part" >&2; bad=1; continue; fi
    size=$(stat -c %s "$dir/$part" 2>/dev/null || stat -f %z "$dir/$part")
    [[ "$size" == "$PART_BYTES" ]] || { echo "size mismatch $part: $size != $PART_BYTES" >&2; bad=1; }
  done
  if [[ -f "$dir/evo2_40b.pt" ]]; then
    size=$(stat -c %s "$dir/evo2_40b.pt" 2>/dev/null || stat -f %z "$dir/evo2_40b.pt")
    [[ "$size" == "$MERGED_BYTES" ]] || { echo "size mismatch evo2_40b.pt: $size != $MERGED_BYTES" >&2; bad=1; }
  fi
  return $bad
}

if [[ "${1:-}" == "--verify-only" ]]; then
  verify "${2:-$DEST}"
  [[ $? -eq 0 ]] && exit 0 || exit 2
fi

mkdir -p "$DEST"
rsync -a --partial --stats "$SRC/" "$DEST/"
status=$?
[[ $status -eq 0 ]] || { echo "rsync failed: $status" >&2; exit $status; }

verify "$DEST" || exit 2

if [[ ! -f "$DEST/evo2_40b.pt" ]]; then
  echo "merging shards..."
  cat "$DEST/evo2_40b.pt.part0" "$DEST/evo2_40b.pt.part1" > "$DEST/evo2_40b.pt" || exit 3
fi
verify "$DEST" || exit 2

sha256sum "$DEST"/evo2_40b.pt.part0 "$DEST"/evo2_40b.pt.part1 "$DEST"/evo2_40b.pt \
  > "$DEST/parts.sha256"
echo "staged to $DEST"
