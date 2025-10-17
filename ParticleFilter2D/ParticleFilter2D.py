import numpy as np

from ParticleFilter2D.Pose2D import Pose2D
from ParticleFilter2D.MotionModel import MotionModel
from ParticleFilter2D.SensorModel import SensorModel

class ParticleFilter2D:
    """2D (i.e. 3-DoF) Particle Filter.

    Implements the 2D particle filtering algorithm covered during CS427
    and in Thrun et al. Ch4 and Ch8.

    Attributes:
        worldsize (float): size of x and y world dimensions
        poses (array): Nx3 numpy array of particle poses where each pose contains
            the pose's [x, y, theta] values, where theta is stored in radians.
        weights (array): numpy array of N importance particle weights
        motionModel (MotionModel): see MotionModel
        sensorModel (SensorModel): see SensorModel
    """

    def __init__(self, num_particles=50, worldsize = 750):
        """Instantiates a new ParticleFilter2D with num_particles and a worldsize.

        Create a new ParticleFilter2D object containing num_particles particles
        and operating of the size worldsize.

        Args:
            num_particles (int): Number of particles to use in the filter 
                (default: 50).
            worldsize (float): Size of the world in x and y dimensions
                (default: 750).
        """
        self.worldsize = worldsize
        self.poses = Pose2D.randomPoses(num_particles, worldsize)
        self.weights = np.ones((num_particles), dtype=float)/num_particles

        self.motionModel = MotionModel()
        self.sensorModel = SensorModel()

    def processFrame(self, odometry, measurements, landmarks):
        """Perform a single predict, update, resample iteration.

        Args:
            odometry (array): 1x3 numpy array of odometry parameters 
                [dx, dy, dtheta] 
            measurements (list): list of SensorMeasurements.
            landmarks (array): Nx2 array of landmark [x, y] location vectors.            
        """
        self.processMotion(odometry)
        self.processSensor(measurements,landmarks)

        self.resample()

    
    def processSensor(self, measurements, landmarks):
        """Process sensor measurements and update weights.

        Performs update step of the particle filtering algorithm.
        
        Args:
            measurements (list): list of SensorMeasurements.
            landmarks (array): Nx2 array of landmark [x, y] location vectors.
        """        
        num_particles = self.poses.shape[0]

        for i in range(num_particles):
            pose = self.poses[i]
            weight = self.sensorModel.likelihood(pose, measurements, landmarks)
            self.weights[i] = weight

        self.weights += 1e-300  # prevent divide-by-zero
        self.weights /= np.sum(self.weights)


    def processMotion(self, odometry):
        """Process odometry by propagating particles using given odometry.

        Performs prediction step of the particle filtering algorithm.
        
        Args:
            odometry (array): 1x3 numpy array of odometry parameters 
                [dx, dy, dtheta] 
        """                
        self.poses = self.motionModel.propagatePoses(self.poses, odometry)

    def resample(self):
        """Perform resampling step.

        Resample particles by drawing M particles with replacement where each 
        particle has a probability of being selected based on its current
        weight.
        """    
        num_particles = self.poses.shape[0]
        new_poses = np.zeros_like(self.poses)

        # Step size and random start
        step = 1.0 / num_particles
        r = np.random.uniform(0, step)
        c = self.weights[0]
        i = 0

        for m in range(num_particles):
            U = r + m * step
            while U > c:
                i += 1
                c += self.weights[i]
            new_poses[m] = self.poses[i]

        self.poses = new_poses
        self.weights = np.ones(num_particles) / num_particles  # resets weights



    def __str__(self):
        rstr = '\n'.join(["x: {}, y: {}, angle: {}, weight: {}".format(p[0], p[1], p[2], w)
            for p,w in zip(self.poses.T, self.weights)])
        return rstr 
