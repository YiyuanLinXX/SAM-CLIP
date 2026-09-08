#!/usr/bin/env bash
set -euo pipefail

model="${1:-vit_b}"
destination_dir="${2:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/volumes/weights}"

case "${model}" in
  vit_b)
    filename="sam_vit_b_01ec64.pth"
    url="https://dl.fbaipublicfiles.com/segment_anything/${filename}"
    expected_sha256="ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
    ;;
  vit_h)
    filename="sam_vit_h_4b8939.pth"
    url="https://dl.fbaipublicfiles.com/segment_anything/${filename}"
    expected_sha256="a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e"
    ;;
  *)
    echo "Usage: $0 {vit_b|vit_h} [destination-directory]" >&2
    exit 2
    ;;
esac

mkdir -p "${destination_dir}"
destination="${destination_dir}/${filename}"
partial="${destination}.partial"

if [[ -f "${destination}" ]] && echo "${expected_sha256}  ${destination}" | sha256sum --check --status; then
  echo "Already downloaded and verified: ${destination}"
  exit 0
fi

echo "Downloading ${model} weights to ${destination}"
curl --fail --location --retry 5 --continue-at - --output "${partial}" "${url}"
echo "${expected_sha256}  ${partial}" | sha256sum --check
mv "${partial}" "${destination}"
echo "Verified: ${destination}"
