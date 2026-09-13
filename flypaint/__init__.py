"""flypaint: a simulated fruit fly brain that listens to music and paints.

Pipeline
--------
audio file -> per-frame features (audio.py)
           -> Poisson stimulation of Johnston's organ neurons (senses.py)
           -> whole-CNS leaky integrate-and-fire simulation (lif.py, graph.py)
           -> descending / mushroom-body population rates (readout.py)
           -> brush commands on a canvas (painter.py)
"""

__version__ = "0.1.0"
