### Monte Carlo Localisation

Simulation of Monte Carlo Localisation (MCL) for a 2D robot, showing particle filtering in action.

**Key Features:**
- **Pose Estimation:** Updated robot pose using odometry with noise via `Pose2D.addOdometry`.
- **Probabilistic Motion Model:** Applied motion propagation to all particles using `MotionModel.propagatePoses`.
- **Landmark Detection:** Computed visible landmarks from the robot’s perspective using `SensorModel.getVisibleLandmarks`.
- **Sensor Likelihood:** Updated particle weights based on sensor measurements using `SensorModel.likelihood`.
- **Particle Resampling:** Performed stochastic universal sampling via `ParticleFilter2D.resample` to generate the next particle generation.
- **Simulation Visualisation:** Displayed real-time particle distributions relative to the true robot pose and landmarks.

**Technical Stack:**
- **Language:** Python
- **Libraries:** NumPy, Matplotlib, Jupyter Notebooks
- **Environment:** Conda

**Project Highlights:**
- Demonstrates how particle filters can estimate robot position in uncertain environments.
- Particles converge around the true robot pose as sensor data is integrated.
- Visual simulation shows red circle for true robot position, white circles for landmarks, and blue/purple particles representing possible poses.
- Video of the simulation illustrates the spread and contraction of particles as the robot moves and senses its environment.
