import numpy as np

class Pose2D:
    """Helper class containing a number static methods for creating and manipulating 2D poses"""

    @staticmethod
    def randomPoses(num_poses, worldsize):
        """Generates a numpy array of random poses

        Args:
            num_poses (int): number of poses to generate
            worldsize  (float): specifies dimensions of the world (i.e. the 
                maximum values for the pose x and y values).
        
        Returns:
            np.array: Nx3 numpy array of generated poses where each pose contains the pose's
                [x, y, theta] values where theta is stored in radians.
        """
        return np.array((np.random.uniform(0.0, worldsize, num_poses),
                np.random.uniform(0.0, worldsize, num_poses),
                np.random.uniform(-np.pi, np.pi, num_poses))).T

    @staticmethod
    def originPoses(num_poses, worldsize):
        """Generates a numpy array of zero poses (i.e. all having the values [0, 0, 0])

        Args:
            num_poses (int): number of poses to generate
            worldsize  (float): specifies dimensions of the world (i.e. the 
                maximum values for the pose x and y values).
        
        Returns:
            np.array: Nx3 numpy array of generated poses where each pose contains the pose's
                [x, y, theta] values where theta is stored in radians.
        """
        return np.tile(
            np.array([worldsize/2.0, worldsize/2.0, np.pi/2]),
            (num_poses,1))


    @staticmethod
    def addOdometry(poses, odometry):
        """Applies odometry to input poses

        Applies the (non-probabilistic / deterministic) kinematic motion model from the
        CS427 particle filter notes by propagating the set of input poses according to 
        the input odometry. 

        Args:
            poses (np.array): Nx3 or 1x3 numpy array of poses where each pose contains the pose's
                [x, y, theta] values where theta is stored in radians.
            odometry (np.array): 1x3 or Nx3 numpy array of odometry parameters [dx, dy, dtheta].

        Returns:
            np.array: Nx3 or 1x3 numpy array of updated poses.
        """        

        # Ensure both are 2D arrays
        single_pose = False
        if poses.ndim == 1:
            poses = np.expand_dims(poses, axis=0)
            single_pose = True

        if odometry.ndim == 1:
            odometry = np.expand_dims(odometry, axis=0)

        if odometry.shape[0] == 1:
            odometry = np.tile(odometry, (poses.shape[0], 1))

        dx = odometry[:, 0]
        dy = odometry[:, 1]
        dtheta = odometry[:, 2]

        theta = poses[:, 2]

        # Transform dx, dy from robot frame to world frame
        x_new = poses[:, 0] + dx * np.cos(theta) - dy * np.sin(theta)
        y_new = poses[:, 1] + dx * np.sin(theta) + dy * np.cos(theta)
        theta_new = poses[:, 2] + dtheta

        # Normalize the new angles
        theta_new = np.array([Pose2D.normaliseAngle(t) for t in theta_new])

        result = np.stack((x_new, y_new, theta_new), axis=1)

        # Return original shape if input was 1D
        return np.squeeze(result) if single_pose else result



                
        


    @staticmethod
    def normaliseAngle(angle):
        """Normalises the input angle to the range ]-pi/2, pi/2]"""        
        if ((angle < np.pi) and (angle >= -np.pi)):
            return angle
        pi2 = np.pi*2
        nangle = angle - (int(angle/(pi2))*(pi2))
        if (nangle >= np.pi):
            return nangle - pi2
        elif (nangle < -np.pi):
            return nangle + pi2
        return nangle 
