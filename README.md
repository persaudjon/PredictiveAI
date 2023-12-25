# PredictiveAI

# Starting State
The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

# Action Space
There are four discrete actions available:

0: do nothing

1: fire left orientation engine

2: fire main engine

3: fire right orientation engine

# Episode Termination
The episode finishes if:

the lander crashes (the lander body gets in contact with the moon);

the lander gets outside of the viewport (x coordinate is greater than 1);

#Reward explained
For each step, the reward:

is increased/decreased the closer/further the lander is to the landing pad.

is increased/decreased the slower/faster the lander is moving.

is decreased the more the lander is tilted (angle not horizontal).

is increased by 10 points for each leg that is in contact with the ground.

is decreased by 0.03 points each frame a side engine is firing.

is decreased by 0.3 points each frame the main engine is firing.

The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.

