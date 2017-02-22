import random
import math

import numpy
import matplotlib.pyplot as pl

class SimpleEnvironment(object):
    
    def __init__(self, herb):
        self.robot = herb.robot
        self.boundary_limits = [[-5., -5.], [5., 5.]]

        # add an obstacle
        table = self.robot.GetEnv().ReadKinBodyXMLFile('models/objects/table.kinbody.xml')
        self.robot.GetEnv().Add(table)

        table_pose = numpy.array([[ 0, 0, -1, 1.0], 
                                  [-1, 0,  0, 0], 
                                  [ 0, 1,  0, 0], 
                                  [ 0, 0,  0, 1]])
        table.SetTransform(table_pose)

        # goal sampling probability
        self.p = 0.2
        self.delta = .1
        self.visualize = True

    def SetGoalParameters(self, goal_config, p=.2, delta=.05, visualize=True):
        self.goal_config = goal_config
        self.p = p
        self.delta = delta
        self.visualize = visualize
        
    # Generate collision free configs
    def GenerateRandomConfiguration(self, goal_config=None):
        # Goal sampling
        if random.random() < self.p:
            return goal_config if goal_config is not None else self.goal_config

        rand_locn = numpy.array(map(lambda (low, up): low + random.random()*(up-low), zip(*self.boundary_limits)))
        while self.HasCollisions(rand_locn):
            rand_locn = numpy.array(map(lambda (low, up): low + random.random()*(up-low), zip(*self.boundary_limits)))
        return rand_locn

    # Euclidean distance
    def ComputeDistance(self, start_config, end_config):
        return math.sqrt(sum(map(lambda (x1, x2): math.pow(x2-x1, 2), zip(start_config, end_config))))

    # Aooly loc to current transform and return new transform
    def ApplyMotion(self, loc):
        new_transform =  self.robot.GetTransform()
        for i in range(len(loc)):
            new_transform[i][3] = loc[i]
        return new_transform

    # Compute delta config in the direction of end_config
    def ComputeDeltaLocation(self, start_config, end_config):
        dist = self.ComputeDistance(start_config, end_config)
        if dist == 0:
            return end_config
        dir_cosines = map(lambda (x1, x2): (float(x2-x1))/dist, zip(start_config, end_config))
        delta_locn = map(lambda (dir_cosine): self.delta*dir_cosine*dist, dir_cosines)
        new_locn = map(lambda (x, d): x+d, zip(start_config, delta_locn))
        return numpy.array(new_locn)

    # Check if new locn has collisions
    def HasCollisions(self, locn):
        collisions = []
        with self.robot.GetEnv():
            # Apply new_locn to current robot transform
            orig_transform = self.robot.GetTransform()
            new_transform = self.ApplyMotion(locn)
            self.robot.SetTransform(new_transform)

            # Check for collision
            bodies = self.robot.GetEnv().GetBodies()[1:]
            collisions = map(lambda body: self.robot.GetEnv().CheckCollision(self.robot, body), bodies)                

            # Set robot back to original transform
            self.robot.SetTransform(orig_transform)
        return reduce(lambda x1, x2: x1 or x2, collisions)

    #TODO Check to see if extended point is within limits
    def Extend(self, start_config, end_config):
        # Compute new location using directional cosines
        new_locn = self.ComputeDeltaLocation(start_config, end_config)

        # Return new_loc as new configuration if no collision else return None
        return None if self.HasCollisions(new_locn) else numpy.array(new_locn)

    def ShortenPath(self, path, timeout=5.0):
        
        # 
        # TODO: Implement a function which performs path shortening
        #  on the given path.  Terminate the shortening after the 
        #  given timout (in seconds).
        #
        return path


    def InitializePlot(self, goal_config):
        self.fig = pl.figure()
        lower_limits, upper_limits = self.boundary_limits
        pl.xlim([lower_limits[0], upper_limits[0]])
        pl.ylim([lower_limits[1], upper_limits[1]])
        pl.plot(goal_config[0], goal_config[1], 'gx')

        # Show all obstacles in environment
        for b in self.robot.GetEnv().GetBodies():
            if b.GetName() == self.robot.GetName():
                continue
            bb = b.ComputeAABB()
            pl.plot([bb.pos()[0] - bb.extents()[0],
                     bb.pos()[0] + bb.extents()[0],
                     bb.pos()[0] + bb.extents()[0],
                     bb.pos()[0] - bb.extents()[0],
                     bb.pos()[0] - bb.extents()[0]],
                    [bb.pos()[1] - bb.extents()[1],
                     bb.pos()[1] - bb.extents()[1],
                     bb.pos()[1] + bb.extents()[1],
                     bb.pos()[1] + bb.extents()[1],
                     bb.pos()[1] - bb.extents()[1]], 'r')
                    
                     
        pl.ion()
        pl.show()
        
    def PlotEdge(self, sconfig, econfig):
        pl.plot([sconfig[0], econfig[0]],
                [sconfig[1], econfig[1]],
                'k.-', linewidth=2.5)
        pl.draw()

