
Experiment details


domain: Deepmind Lab seekavoid-arena0
description: Agent must navigate a 3 dimensional arena, collecting apples for positive reward and avoiding lemons that
yield negative reward.

agent details and notes:
A3C agent with an unspecified number of threads. The aux reward paper uses 32 but this exceeds the number of
diae cores (6) so likely it will be run with 6. This may hurt stability. Updates happen every 20 environment steps. Network
architecture is that of the auxiliary tasks paper (2 convolutional, 1 fc, 1 lstm, 1 linear to outputs).


description of results:
