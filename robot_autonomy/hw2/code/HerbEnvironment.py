import math
import random
import numpy

class HerbEnvironment(object):
    
    def __init__(self, herb):
        self.robot = herb.robot

        # add a table and move the robot into place
        table = self.robot.GetEnv().ReadKinBodyXMLFile('models/objects/table.kinbody.xml')
        self.robot.GetEnv().Add(table)

        table_pose = numpy.array([[ 0, 0, -1, 0.6], 
                                  [-1, 0,  0, 0], 
                                  [ 0, 1,  0, 0], 
                                  [ 0, 0,  0, 1]])
        table.SetTransform(table_pose)

        # set the camera
        camera_pose = numpy.array([[ 0.3259757 ,  0.31990565, -0.88960678,  2.84039211],
                                   [ 0.94516159, -0.0901412 ,  0.31391738, -0.87847549],
                                   [ 0.02023372, -0.9431516 , -0.33174637,  1.61502194],
                                   [ 0.        ,  0.        ,  0.        ,  1.        ]])
        self.robot.GetEnv().GetViewer().SetCamera(camera_pose)
        
        # goal sampling probability
        self.p = 0.0
        self.delta = 0.01
        self.visualize = False

    def SetGoalParameters(self, goal_config, p = 0.2, visualize = False, delta = .05):
        self.goal_config = goal_config
        self.p = p
        self.visualize = visualize
        self.delta = delta
        
    def GenerateRandomConfiguration(self, goal_config=None):
        if random.random() < self.p:
            return goal_config if goal_config is not None else self.goal_config

        rand_conf = numpy.array(map(lambda (low, up): low + random.random()*(up-low), zip(*self.robot.GetActiveDOFLimits())))
        while (self.HasCollisions(rand_conf)):
            rand_conf = numpy.array(map(lambda (low, up): low + random.random()*(up-low), zip(*self.robot.GetActiveDOFLimits())))
        return rand_conf

    def ComputeDistance(self, start_config, end_config):
        return math.sqrt(sum(map(lambda (x1, x2): math.pow(x2-x1, 2), zip(start_config, end_config))))

    def ComputeDeltaLocation(self, start_config, end_config):
        dist = self.ComputeDistance(start_config, end_config)
        dir_cosines = map(lambda (x1, x2): (float(x2-x1))/dist, zip(start_config, end_config))
        delta_locn = map(lambda (dir_cosine): self.delta*dir_cosine*dist, dir_cosines)
        new_dof_vals = map(lambda (x, d): x+d, zip(start_config, delta_locn))
        return numpy.array(new_dof_vals)

    def HasCollisions(self, new_dof_vals):
        collisions = []
        with self.robot.GetEnv():
            # Apply new_locn to current robot transform
            orig_active_dof_vals = self.robot.GetActiveDOFValues()
            self.robot.SetActiveDOFValues(new_dof_vals)

            # Check for collision
            bodies = self.robot.GetEnv().GetBodies()[1:]
            collisions = map(lambda body: self.robot.GetEnv().CheckCollision(self.robot, body), bodies)                

            # Set robot back to original transform
            self.robot.SetActiveDOFValues(orig_active_dof_vals)

        # Return new_loc as new configuration if no collision else return None
        return reduce(lambda x1, x2: x1 or x2, collisions)

    #TODO Check to see if extended point is within limits
    def Extend(self, start_config, end_config):
        # Compute new location using directional cosines
        new_dof_vals = self.ComputeDeltaLocation(start_config, end_config)
      
        # Return new_loc as new configuration if no collision else return None
        return None if self.HasCollisions(new_dof_vals) else numpy.array(new_dof_vals)
        
    def ShortenPath(self, path, timeout=5.0):
        
        # 
        # TODO: Implement a function which performs path shortening
        #  on the given path.  Terminate the shortening after the 
        #  given timout (in seconds).
        #
        return path
