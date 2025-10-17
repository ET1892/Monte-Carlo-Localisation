import numpy as np
import scipy.stats

from ParticleFilter2D.Pose2D import Pose2D

class SensorMeasurement:
    """Simple class for representing a (range, bearing) measurement to a specified landmark."""

    def __init__(self, range, bearing, landmark_index):
        """ Initialises a SensorMeasurement.

        Args:
            range (float): distance/range to landmark
            bearing (float): bearing angle to landmark in radians
            landmark_index (int): index of landmark in corresponding map
        """
        self.range = range
        self.bearing = bearing
        self.landmark_index = landmark_index

    def __str__(self):
        rstr = "r: {}, b: {}, i: {}".format(self.range, self.bearing, self.landmark_index)
        return rstr 

class SensorModel:
    """ Probabilistic Sensor Model based on example from CS427 Particle Filter Notes.

    Attributes:
        max_range (float): maximum sensing range.
        min_range (float): minimum sensing range.
        fov (float): field of view of the sensor in radians. Technically the 
            field of view of the sensors is 2*fov i.e. it has a field of view
            from -fov..+fov radians.
        rangeStd (float): standard deviation of range measurements.
        bearingStd (float): standard deviation of bearing measurements.
    """

    def __init__(self, min_range = 10.0, max_range = 150.0, fov = 0.523598775,
                    rangeVar = 10.0, bearingVar = 0.087266462):
        """ Initialises SensorModel with default or provided variances

        Args:
            min_range (float): minimum sensing range (default: 10.0).
            max_range (float): maximum sensing range (default: 150.0).
            fov (float): field of view of the sensor in radians. Technically the 
                field of view of the sensors is 2*fov i.e. it has a field of view
                from -fov..+fov radians. (default: 0.523598775 radians (i.e. 30deg))
            translationVar (float): range error *variance* i.e. this is equal to
                the stddev^2 (default: 10.0).
            bearingVar (float): bearing error *variance* in radians i.e. this is equal to
                the stddev^2 (default: 0.087266462 (i.e. 5 deg)).
        """
        self.max_range = max_range
        self.min_range = min_range
        self.fov = fov

        self.rangeStd = np.sqrt(rangeVar)
        self.bearingStd = np.sqrt(bearingVar)        

    def __senseAllLandmarks(self, pose, landmarks):
        """Exhaustively calculates the (range, bearing) measurements to all landmarks.
        
        Args:
            pose (array): 1x3 array of pose [x, y, theta] values where theta is 
                stored in radians.
            landmarks (array): Nx2 array of landmark [x, y] location vectors.

        Returns:
            ranges (array): array of ranges from pose to each landmark.
            angles (array): array of angles from the pose to each landmark.
        """
        rep_pose = np.tile(pose, (landmarks.shape[0], 1))
        linesOfSight = landmarks - rep_pose[:,0:2]
        ranges = np.linalg.norm(linesOfSight, axis=1)
        anglesOfSight = (np.arctan2(linesOfSight[:,1], linesOfSight[:,0]))
        angles = anglesOfSight - rep_pose[:,2]
        return ranges, angles
        
    def getVisibleLandmarks(self, pose, landmarks):
        """Returns an array of SensorMeasurements to visible landmarks."""
        ranges, angles = self.__senseAllLandmarks(pose, landmarks)

        visible_landmarks = []

        for i in range(len(landmarks)):
            r = ranges[i]
            a = angles[i]

            if self.min_range <= r <= self.max_range and -self.fov <= a <= self.fov:
                visible_landmarks.append(SensorMeasurement(r, a, i))

        return visible_landmarks

    def sampleMeasurements(self, pose, landmarks):
        """Returns an array of *noise corrupted* measurements to visible landmarks.

        Applies the probabilistic sensor model from CS427 particle filter notes
        by computing a set of SensorMeasurements including (range, bearing)
        that include the addition of Gaussian noise with the noise parameters
        stored in the corresponding member variables. Note: that measurements are
        only returned for landmarks that are within the visible range specified
        by the associated member variables.

        Args:
            pose (array): 1x3 array pose [x, y, theta] values where theta is
                stored in radians.
            landmarks (array): Nx2 array of landmark [x, y] location vectors.

        Returns:
            visible_landmarks (list): list of *noise corrupted* SensorMeasurements
                to visible landmarks.
        """        
        visible_landmarks = self.getVisibleLandmarks(pose, landmarks)

        measurement = lambda m : SensorMeasurement(
                m.range + np.random.normal(loc=0, scale=self.rangeStd), 
                Pose2D.normaliseAngle(m.bearing + np.random.normal(loc=0, scale=self.bearingStd)), 
                m.landmark_index) 

        visible_landmarks = [measurement(m) for m in visible_landmarks]
        
        return visible_landmarks


    def likelihood(self, pose, measurements, landmarks):
        """Returns the likelihood of a set measurements given the pose and landmarks.

        Evaluates the measurement likelihood model described in the CS427
        particle filter notes. This computes the likelihood of each measurement
        by evaluating the product of (i) the Gaussian of the deviation in the 
        range, and, (ii) the Gaussian of the deviation in the bearing. The
        complete likelihood is given by the product of the likelihood of each
        of the individual measurements.
        """
        ranges, angles = self.__senseAllLandmarks(pose, landmarks)
        q = 1

        ## STEP 4: Add the code here required to compute the likelihood of the
        ##          measurements given the pose and the landmarks
        ## Hint: it may be wortwhile to look at scipy.stats.norm.pdf

        for m in measurements:
            idx = m.landmark_index
            expected_range = ranges[idx]
            expected_bearing = angles[idx]

            p_range = scipy.stats.norm.pdf(m.range, loc=expected_range, scale=self.rangeStd)
            p_bearing = scipy.stats.norm.pdf(m.bearing, loc=expected_bearing, scale=self.bearingStd)

            q *= p_range * p_bearing

        return q
