# Executive summary (read this first)

Test the actual Docker image on every published card in a fresh restricted
container before selecting a Final image. Check real HTTP transport with a local
synthetic proxy; no organizer House access or outcome is involved. A passing run
establishes execution only, not improved forecasts or official attestation.

## Reproduce the check

Use the pinned Python 3.13/common-2.6.0 environment, Docker for Linux containers,
and an unmodified official kit at the revision named in the workflow. From this
checkout:

```bash
docker build --platform linux/amd64 -t t2-final-runtime .
OPENBLAS_NUM_THREADS=1 python -m scripts.check_final_container \
  --image t2-final-runtime --units /absolute/official-kit/units \
  --out /absolute/private/new-runtime-check
```

The driver refuses to put outputs in this checkout or overwrite a previous run.
It verifies the organizer's input checksums and stages input-only snapshots,
excluding outcomes and reference scales. Original kit files are never changed.
It records the exact local image ID and a private per-container runtime report.
The research workflow builds locally; it does not publish an image, package a ZIP
or upload to CodaBench. Its artifact contains the runtime report only.

## Restrictions actually exercised

Each card gets a new ordinary Docker container. It runs as uid/gid 65534 without
a home directory, with a read-only root and input, no capabilities, no privilege
escalation and no external network. `/tmp` is 64 MiB with `noexec,nosuid,nodev`.
Limits are 256 processes/threads, 1024 file descriptors and 64 MiB per file.
The driver also checks that the complete output tree is at most 64 MiB and has
exactly the three standard forecast files. The unchanged canonical output gates
validate each result without accessing outcomes.

The 2-CPU/2-GiB limits are conservative test settings, not a statement of Final
resource grants. Ordinary Docker does not reproduce an organizer sandbox,
provisioning delays, image pulls, total-roster timing or live model latency.
The image must be Linux/amd64, have interface label 2.0 and declare no VOLUME.

## HTTP and fallback cases

Every card first runs the real `forecast` command without model configuration.
One daily card from each of T2-F1 through T2-F4 additionally exercises:

- the numerical CLI as a fixed reference;
- successful extraction and scenario requests through a real loopback HTTP proxy;
- the same successful transport with event timing enabled;
- HTTP 503, malformed JSON and a response truncated by its token limit.

The proxy checks the absolute `/v1/chat/completions` URL, bearer and proxy
authentication headers, injected model name, output-token limit and thinking
setting. Its replies quote the supplied corpus for an interface test only.
Successful replies must change exactly half the joint paths. Every failure
case must retain the numerical forecast bit for bit. Synthetic replies do not
establish forecast quality and cannot be used to select a text model.

## House access and the remaining quality decision

The official [runtime guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/docs/DEVELOPMENT-RUNTIME.md)
describes a participant-owned compatible server for local tests. The
[House guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/docs/HOUSE-MODEL.md)
describes credentials injected per official unit. These do not supply reusable
off-platform House credentials. No live House credential is available to this
research checkout. A different local model would provide proxy evidence only,
not measured performance of the approved service.

The new short-horizon kit is still awaited on 14 October. Use it for genuine
numeric validation and release-card integration when published. Actual House
quality comparisons require an authorized testing opportunity; they are not
guaranteed to become available before the one Final upload. Keep that limitation
explicit when making the final model choice.
