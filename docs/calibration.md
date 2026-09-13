# Calibration of the MaleCNS spiking model

## The problem

With the published Shiu et al. 2024 constants (0.275 mV per synaptic contact,
tau_m 20 ms, threshold 7 mV above rest) the unmodified MaleCNS graph is not in a
responsive regime. Stimulating **114 random neurons at 50 Hz** ignites a
self-sustaining storm within 60 ms:

```
114 random neurons at 50 Hz, all 25.6 M edges
  t= 20 ms  spikes    123  active downstream      8
  t= 40 ms  spikes    535  active downstream    294
  t= 60 ms  spikes  24131  active downstream   9705
  t= 80 ms  spikes  59175  active downstream  15946
  t=100 ms  spikes  65990  active downstream  16752   (steady state: ~17k neurons at ~400 Hz)
```

The same happens with the 5-synapse connection threshold of the FlyWire release the
original model used (6.2 M edges). The reason is data, not code: MaleCNS reports
124 M synaptic contacts among 165 k traced neurons (about 750 per neuron), roughly
twice FlyWire's density, and the mean excitatory drive per neuron is 128 mV of
summed incoming weight against 79 mV of inhibition.

## Sweep

`python scripts/calibrate.py --min-synapses 5 --ms 300`

Four 300 ms trials per configuration: auditory JONs (114 cells) at 200 Hz, a
size-matched random set at 200 Hz, wind/gravity JONs (475 cells) at 100 Hz.
`aud_act` = downstream neurons active, `late/early` = downstream spike rate in the
last 100 ms over the first 200 ms (>1 means still growing), `GF_hz` = giant fibre
DNp01 rate under auditory drive, `jac` = Jaccard overlap of the 25 top-responding
cell types between the auditory and wind trials (0 = fully stimulus-specific).

```
config                                    aud_act  aud_hz late/early  ctl_act  ctl_hz wind_act  GF_hz jac(aud,wind)
w_gain=1.0                                  21121  10.882       1.29    20394  11.000    21248    5.0          0.09
w_gain=0.5                                    454   0.087       1.23    10019   3.879     6265   70.0          0.00
w_gain=0.35                                   186   0.044       1.20     5319   1.765     1246   81.7          0.00
w_gain=0.25                                    94   0.026       1.20     2661   0.468      308   73.3          0.09
w_gain=1.0,adapt_mv=4.0                     22505   4.150       0.61    21876   4.171    23830    5.0          0.85
w_gain=1.0,adapt_mv=8.0                     19988   2.215       0.49    19779   2.396    22238   30.0          0.85
w_gain=0.5,adapt_mv=4.0                       370   0.028       0.61     7389   1.214     1775   55.0          0.11
w_gain=0.5,adapt_mv=8.0                       295   0.017       0.48     6615   0.779     1226   35.0          0.09
w_gain=0.35,adapt_mv=4.0                      158   0.015       0.71     3995   0.532      453   41.7          0.11
w_gain=0.35,adapt_mv=8.0                      151   0.009       0.57     3542   0.324      423   25.0          0.11
w_gain=1.0,std_u=0.2                         6081   0.384       0.01    11376   0.937    11808   15.0          0.00
w_gain=1.0,std_u=0.4                         3595   0.165       0.00     6097   0.405     2458   10.0          0.04
w_gain=0.5,std_u=0.2                           75   0.004       0.04     1749   0.209      381   16.7          0.09
w_gain=0.5,std_u=0.4                           50   0.002       0.00        7   0.000      229    5.0          0.09
w_gain=0.5,adapt_mv=4.0,std_u=0.2              71   0.002       0.00     1539   0.093      342    6.7          0.06
w_gain=0.35,adapt_mv=4.0,std_u=0.2             33   0.001       0.00        7   0.000      171    5.0          0.09
```

Reading the table:

* Gain 1.0 with any stabiliser is still a storm (20 k active, top cell types
  identical for sound and wind).
* Gain alone at 0.5 or below is bounded and stimulus-specific, and the JO-A to giant
  fibre pathway (Kim et al. 2020) lights up strongly. But activity is still growing at
  300 ms (late/early > 1), so a long loud passage could tip over.
* Adding 4 mV spike-frequency adaptation (200 ms decay) turns that into a decaying
  response (late/early 0.6) while keeping specificity and the giant-fibre response.
* Short-term depression at these strengths quenches everything within 100 ms, which is
  the wrong behaviour for a brain that has to keep listening to a song.
* The random control recruits more neurons than the auditory set at every gain. That
  is expected: a random draw includes central hub neurons with thousands of outputs,
  while sensory JONs fan out to a small second-order population.

## Defaults

`w_gain = 0.5`, `adapt_mv = 4.0`, `adapt_tau_ms = 200`, `min_synapses = 5`.
These are engineering choices that keep the wiring diagram in a responsive regime;
they are not measured fly physiology. `flypaint validate` re-checks them.
