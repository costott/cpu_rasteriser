#!/usr/bin/env bash
# Renders silicon_01 offline at full quality and encodes it with ffmpeg (which must be on PATH):
# the whole loop as a video with its soundtrack, or the README's highlight GIF, which is silent.
# Run it from the workspace root.
#
#   engine/examples/silicon_01/media.sh video <out.mp4> [fps=60] [size=1920x1080]
#   engine/examples/silicon_01/media.sh gif <out.gif> [fps=10] [width=560]
#   engine/examples/silicon_01/media.sh audio <video.mp4> <out.mp4>
#
# `audio` renders the soundtrack and puts it on an existing video of the whole loop, copying its
# picture as it is, so the sound can be changed without recording the frames again.
#
# Every frame is rendered at twice the output's resolution and filtered down (--scale 2). Frames
# are recorded in short spans under target/silicon_01_media/<mode>_<size>_<fps>fps/. A finished
# span is kept until the example is rebuilt, so an interrupted run carries on where it stopped.
# Frames are saved at the window's size, so the screen must be able to hold a window of that size,
# and the windows must be left alone: if one is resized or minimised while it records, delete that
# span's directory and run the script again.
set -euo pipefail

example=engine/examples/silicon_01
[ -f "$example/main.rs" ] || { echo "run this from the workspace root" >&2; exit 1; }

# Highlights for the GIF as "from to" in seconds (see director.rs for the acts), recorded without
# captions: the die on its own, turning; the teapot building; the mandelbulb turning; the eruption;
# then the first seven worlds and the montage's strobe. About 25 s.
gif_segments=(
  "28.6 31.1"
  "38 41.5"
  "51 54.5"
  "66 67.8"
  "69.3 80.9"
  "98 100.5"
)
span_length=10

usage="usage: media.sh video|gif <out> [fps] [size or width], or media.sh audio <video> <out>"
mode=${1:?$usage}
out=${2:?$usage}
# Encodes the sound alongside the picture.
sound_codec=(-c:a aac -b:a 256k)
case "$mode" in
  video) fps=${3:-60}; size=${4:-1920x1080}; flags=() ;;
  gif) fps=${3:-10}; width=${4:-560}; size=1280x720; flags=(--no-captions) ;;
  audio) video=$out; out=${3:?$usage} ;;
  *) echo "unknown mode $mode: $usage" >&2; exit 1 ;;
esac

# One job keeps the build's peak memory to a few gigabytes.
cargo build --release -j 2 -q -p engine --example silicon_01
exe=target/release/examples/silicon_01
[ -f "$exe.exe" ] && exe=$exe.exe

# Renders the soundtrack of the whole loop to the file named.
soundtrack() {
  mkdir -p "$(dirname "$1")"
  "$exe" --soundtrack "$1"
}

if [ "$mode" = audio ]; then
  wav=target/silicon_01_media/soundtrack.wav
  soundtrack "$wav"
  ffmpeg -y -loglevel error -i "$video" -i "$wav" -map 0:v -map 1:a -c:v copy "${sound_codec[@]}" \
    -movflags +faststart "$out"
  ls -l "$out"
  exit
fi

work=target/silicon_01_media/${mode}_${size}_${fps}fps
stamp=$(sha1sum "$exe" | cut -c1-16)
if [ "$(cat "$work/build" 2>/dev/null)" != "$stamp" ]; then
  rm -rf "$work"
  mkdir -p "$work"
  echo "$stamp" >"$work/build"
fi

# Records (from, to] into its own directory: the clock steps before the first frame is drawn, so
# consecutive spans join without repeating a frame.
record() {
  local dir="$work/span_$1_$2"
  [ -f "$dir/done" ] && return
  rm -rf "$dir"
  echo "recording $1–$2 s"
  "$exe" --record "$dir" --from "$1" --to "$2" --fps "$fps" --window "$size" --scale 2 \
    "${flags[@]}" >/dev/null
  touch "$dir/done"
}

# Links the recorded spans' frames, in order, into one numbered sequence for ffmpeg.
collect() {
  local all="$work/all" i=0 span f
  rm -rf "$all"
  mkdir -p "$all"
  for span in "$@"; do
    for f in "$work/span_${span% *}_${span#* }"/frame_*.png; do
      ln -f "$f" "$all/$(printf 'f_%05d.png' "$i")"
      i=$((i + 1))
    done
  done
  echo "$i frames"
}

spans=()
if [ "$mode" = video ]; then
  loop=$(sed -n 's/^pub const LOOP_LENGTH: f32 = \([0-9.]*\);/\1/p' "$example/director.rs")
  for ((from = 0; from < ${loop%.*}; from += span_length)); do
    to=$((from + span_length))
    [ "$to" -gt "${loop%.*}" ] && to=$loop
    spans+=("$from $to")
  done
else
  spans=("${gif_segments[@]}")
fi
for span in "${spans[@]}"; do
  record ${span}
done
collect "${spans[@]}"

frames="$work/all/f_%05d.png"
if [ "$mode" = video ]; then
  soundtrack "$work/soundtrack.wav"
  ffmpeg -y -loglevel error -framerate "$fps" -i "$frames" -i "$work/soundtrack.wav" \
    -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p "${sound_codec[@]}" \
    -movflags +faststart "$out"
else
  filters="scale=$width:-1:flags=lanczos"
  ffmpeg -y -loglevel error -framerate "$fps" -i "$frames" \
    -vf "$filters,palettegen=max_colors=192:stats_mode=diff" "$work/palette.png"
  ffmpeg -y -loglevel error -framerate "$fps" -i "$frames" -i "$work/palette.png" \
    -lavfi "$filters [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
    -loop 0 "$out"
fi
ls -l "$out"
