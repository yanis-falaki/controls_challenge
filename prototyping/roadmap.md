# Roadmap
High level goal: Pretrain MLP on PID outputs, fine-tune with mlp

- Generate a bunch of runs with PID, save state, action, and cost

# Thoughts
I'm thinking I should perform attention on the past states and lataccels and possibly targets.
This could possibly reveal some hidden dynamics in the system.

I should also perform attention on the forward looking targets as well, as they may say something of the upcoming topology.

I'm not sure if I want to do attention all together with both the past and the future targets. Immediately I think no as what does the past topology necessarilly have to do with the future, but there may be some correlations so I'm not sure.

So right now both my policy and critic models would have the following:

A segment (A) which does self attention on the 50 past states + lat_accel + target_accel
    I probably will want an mlp which turns state + lat_accel + target_lat_accel into an embedding, then do attention on those + positional_embedding

A segment (B) which does self attention on the 50 next target_lataccel

An mlp which takes the output from segment A, segment B, current_state, current_lataccel, and target_lataccel, that outputs a steering action in [-2, 2]