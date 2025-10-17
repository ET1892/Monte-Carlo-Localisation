import numpy as np

from ParticleFilter2D.Pose2D import Pose2D

class MotionModel:
    """ Probabilistic Motion Model based on example from CS427 Particle Filter Notes.

    Attributes:
        translationStd: standard deviation of translation error used for both x and y
        rotationStd: standard deviation of rotation error
    """

    def __init__(self, translationVar=0.02, rotationVar=0.0174532924):
        """ Initialises MotionModel with default or provided variances

        Args:
            translationVar (float): translation error *variance* i.e. this is equal to
                the stddev^2. (default: 0.02)
            rotationVar (float): rotation error *variance* in radians i.e. this is equal to
                the stddev^2 (default: 0.0174532924 rads (i.e. 1 deg))
        """
        self.translationStd = np.sqrt(translationVar)
        self.rotationStd = np.sqrt(rotationVar)

    def propagatePoses(self, poses, odometry):
        """ Probabilistically applies input odometry to propate the given poses

        Applies the probabilistic motion model from CS427 particle filter notes by propagating
        the set of input poses according to the input odometry, where the motion includes
        the addition of Gaussian noise with the noise parameters stored in the corresponding
        member variables. -> Add Gaussian Noise

        Args:
            poses (np.array): Nx3 numpy array of poses where each pose
                contains the pose's [x, y, theta] values where theta is stored
                in radians.
            odometry (np.array): 1x3 numpy array of odometry parameters 
                [dx, dy, dtheta] 

        Returns:
            np.array: Nx3 numpy array of updated poses where each pose contains the pose's
                [x, y, theta] values where theta is stored in radians.  
        """  
        npts = poses.shape[0]
        odometry = np.tile(odometry, (npts, 1))

        ## STEP 3: Currently this method propagates the particle
        ##          set without noise. Alter this function such that 
        ##          it implements the complete motion model (i.e. 
        ##          including noise), as discussed in the notes

        # Add Gaussian noise to odometry
        noisy_dx = odometry[:, 0] + np.random.normal(0, self.translationStd, npts)
        noisy_dy = odometry[:, 1] + np.random.normal(0, self.translationStd, npts)
        noisy_dtheta = odometry[:, 2] + np.random.normal(0, self.rotationStd, npts)

        noisy_odometry = np.stack((noisy_dx, noisy_dy, noisy_dtheta), axis=1)

        poses = Pose2D.addOdometry(poses, noisy_odometry)
        return poses

