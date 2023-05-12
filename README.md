# 🤣rofl-beta

ROFL is an improved version of LLOL.
It makes ues of two panos instead of one for odometry estimation.
Using two panos eliminates the need for the costly step of moving the pano to a new location.
Instead, it simply initializes a new pano at the desired location and add new sweeps to it, while the old pano can be used for odometry estimation.


## Reference

LLOL: Low-Latency Odometry for Spinning Lidars

Chao Qu, Shreyas S. Shivakumar, Wenxin Liu, Camillo J. Taylor

https://arxiv.org/abs/2110.01725

https://youtu.be/MmiTMFt9YdU

https://www.youtube.com/watch?v=VwP1JMlxOgc

https://github.com/versatran01/llol
