# flypaint

A simulated male fruit fly brain listens to music and paints its reaction.

The brain is the complete **MaleCNS v1.0** connectome (Janelia FlyEM, Cambridge and
Google Research, 2026): 165,122 traced neurons and 25.6 million directed connections
covering the brain and ventral nerve cord of one adult male *Drosophila melanogaster*.
It is run as a leaky integrate-and-fire spiking network with the constants from
Shiu et al. 2024 (*Nature*), the same recipe behind Stonkfly, DOOMFLY and the other
MaleCNS demos.

```
song.wav ─► antennal features ─► Johnston's organ neurons ─► 165k-neuron LIF brain
                                                                  │
   painting.png ◄── brush (walk, turn, colour, width, splat) ◄── descending neurons,
                                                                  mushroom-body outputs,
                                                                  dopamine neurons,
                                                                  giant fibre
```

## What is real and what is engineered

**Real:** every neuron, every connection and its synapse count, the neurotransmitter
predictions that set each connection's sign, the spiking neuron model and its
constants, and the identity of the sensory and output neurons.

**Engineered:** how sound becomes firing rates, and how firing rates become paint.

*Ears.* Flies hear with their antennae. Johnston's organ neurons (JONs) come in groups
with known tuning: JO-A (sound, broad, peak ~400 Hz), JO-B (sound, below ~200 Hz,
the courtship pulse song band), JO-C/E (static deflection: wind and gravity) and
JO-D/F (antennal touch and grooming). MaleCNS labels its 672 JONs with those
subclasses and a left/right side. Per 10 ms audio frame we compute band energies in
the JO-A and JO-B bands, a slow loudness envelope and an onset strength, and drive the
matching JON groups as Poisson spike sources. The left channel drives the left antenna.

One caveat from the data: in MaleCNS v1.0 most right-side auditory JONs have no
traced outputs (24 of 31 right JO-A cells, 15 of 19 right JO-B cells), so the released
fly is nearly deaf on the right. By default the left antenna therefore hears the mono
mix and the right antenna hears the right channel (`--ear stereo` keeps a strict
left-to-left, right-to-right mapping).

*Hands.* The brush is the fly walking on the canvas:

| channel | source | why |
|---|---|---|
| turn | right minus left mean rate of all descending neurons, high-passed | turning is a left/right asymmetry of descending drive to the legs |
| speed | mean rate of all 1,314 descending neurons | descending drive to the legs |
| hue | neurons selective for the JO-B low band vs the JO-A high band | the brain's own pitch axis (low = deep blue, high = warm yellow) |
| saturation | neurons selective for antennal touch (transients) | onsets make the ink vivid |
| value | escape descending neurons DNp01, DNp02, DNp11 | startle darkens the ink |
| brush width | total downstream spike rate | brain-wide arousal, dominated by the wind/gravity population |
| ink splat | giant fibre DNp01 spike (250 ms refractory) | the escape jump |

The "selective" populations are measured, not hand-picked: each ear channel is driven
alone at full rate and the neurons that respond at least twice as strongly to it as to
any other channel form its signature (`flypaint/signatures.py`). The mushroom body,
central complex and dopamine neurons stay silent under antennal input in this
connectome, which is why colour is not read from them.

Nothing here claims the fly "likes" a song. It is a faithful wiring diagram driven by
real sound and read out through neurons with documented roles or measured responses.

## Quickest start: Docker

```bash
git clone https://github.com/namdar12/flyPaint.git && cd flyPaint
docker compose up
```

Open http://localhost:8000. The first start downloads the 1.1 GB connectome into
`./data` and builds the graph cache (under a minute); later starts are instant.
Paintings land in `./runs/web`. Both folders are plain host directories, so nothing is
lost when the image is rebuilt. The container needs about 2 GB of memory (the one-time
cache build peaks at 1.7 GB, painting uses 0.5 GB); Docker Desktop's default is enough.

Other CLI commands run through the same image:

```bash
docker compose run --rm flypaint validate
docker compose run --rm flypaint paint /app/runs/song.mp3 -o /app/runs/song
```

## Web app

`flypaint serve` (or the Docker container) hosts a local single-page app:

* drop in a wav, flac, ogg or mp3
* every setting in the five groups (run, brain, ears, readout, painter) is a slider
  with its help text; settings persist in the browser and can be copied back from
  any earlier painting with "Use these settings"
* jobs queue on one worker thread with the connectome loaded once; a live preview
  refreshes every two seconds while the fly paints, with the current descending
  rate, giant-fibre rate, brush command, ink colour and the four ear channels
* finished paintings show with their timelapse and downloads for the PNG, GIF,
  per-frame log and metadata, and stay in a gallery across restarts

The API is plain JSON under `/api/` (`/api/settings` for the schema, `/api/jobs` to
submit and list, `/api/jobs/<id>/painting.png` and friends for files).

### Listen live (Spotify, YouTube, anything in a tab)

Streaming services never expose their audio, so instead the fly listens to what your
browser is playing. Press **Listen live**, pick the tab that is playing (Spotify's web
player at open.spotify.com works) and tick *Share tab audio*. The page streams the raw
sound to the container over a WebSocket, and the brain paints as it arrives, with the
canvas, lag and neuron rates updating live. **Stop & save** writes the run to the
gallery like any other painting, with the captured audio alongside it.

Details worth knowing:

* Chrome and Edge can share a tab's audio; Safari and Firefox cannot. On Windows,
  sharing the entire screen with system audio captures the desktop Spotify app too.
* Live mode defaults to 1 ms of brain time per 20 ms of audio, about five times
  less brain time than a file run, so that a 4-core laptop keeps up in real time.
  If the brain still falls more than 1.5 s behind, audio frames are skipped and the
  count is shown; untick *Real-time brain speed* to use the Run settings instead.
* Features are normalised against a running level rather than the whole track, so the
  first few seconds of a session are calibrating.
* `scripts/live_client.py` streams a file to the endpoint at real-time pace, for
  testing the live path without a browser.

## Install without Docker

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[test,web]'
flypaint prepare          # downloads ~1.1 GB of MaleCNS files, builds the graph cache (<1 min)
flypaint validate         # sanity checks against a random-stimulation control
flypaint bench            # how fast is this machine?
flypaint serve            # web app on http://127.0.0.1:8000
flypaint settings         # every tunable setting with defaults and help
```

`prepare` fetches three public files from `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`
(connection weights, body annotations, neurotransmitter predictions). The edge file is
streamed in batches, so building the cache peaks at 1.7 GB of RAM and painting uses 0.5 GB.

## Paint

```bash
flypaint demo-song runs/demo.wav --seconds 30     # synthetic test track (pad, pulse song, drums)
flypaint paint runs/demo.wav -o runs/demo
flypaint paint ~/Music/track.mp3 --max-seconds 60
flypaint paint song.wav --set hue_low=0.55 --set opacity=0.6 --set max_step_px=9
```

Any of the 35 settings listed by `flypaint settings` can be passed with `--set`.

Outputs in the run directory: `painting.png`, `timelapse.gif`, `frames/`, a per-frame
`log.jsonl` with every readout rate and brush command, and `meta.json`.

The brain clock is compressed: by default each 10 ms of audio drives 2 ms of brain
time (`--brain-ms`). Real-time (`--brain-ms 10`) is five times slower and looks
similar because the sensory features are already smoothed at the frame level.

## Simulator notes

* NumPy only. Spikes are propagated with a vectorised gather over the CSR adjacency, so
  cost scales with activity, not with the 25.6 M edges. A quiet brain runs at about
  3 s per brain-second on 4 CPU cores; a loud one at 10-15 s.
* MaleCNS reports roughly twice the synaptic contacts per neuron of FlyWire. With the
  published 0.275 mV per contact the unmodified network ignites into a self-sustaining
  storm from almost any input (see `scripts/calibrate.py`). The simulator therefore
  offers three standard stabilisers (global weight gain, spike-frequency adaptation,
  short-term synaptic depression); the defaults come from the calibration sweep in
  `docs/calibration.md`.
* `--min-synapses 5` reproduces the connection threshold of the FlyWire release the
  original model was built on.

## Data credits

MaleCNS v1.0 connectome: FlyEM Project Team, HHMI Janelia Research Campus; Jefferis
lab, MRC LMB / University of Cambridge; Google Research. https://male-cns.janelia.org
Neuron model: Shiu, Sterne et al., "A Drosophila computational brain model reveals
sensorimotor processing", *Nature* 2024.
