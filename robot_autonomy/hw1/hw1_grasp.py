#!/usr/bin/env python


PACKAGE_NAME = 'hw1'

# Standard Python Imports
import os
import copy
import time
import math
import numpy as np
np.random.seed(0)
import scipy

# OpenRAVE
import openravepy
#openravepy.RaveInitialize(True, openravepy.DebugLevel.Debug)


curr_path = os.getcwd()
relative_ordata = '/models'
ordata_path_thispack = curr_path + relative_ordata



#this sets up the OPENRAVE_DATA environment variable to include the files we're using
openrave_data_path = os.getenv('OPENRAVE_DATA', '')
openrave_data_paths = openrave_data_path.split(':')
if ordata_path_thispack not in openrave_data_paths:
  if openrave_data_path == '':
      os.environ['OPENRAVE_DATA'] = ordata_path_thispack
  else:
      datastr = str('%s:%s'%(ordata_path_thispack, openrave_data_path))
      os.environ['OPENRAVE_DATA'] = datastr

#set database file to be in this folder only
relative_ordatabase = '/database'
ordatabase_path_thispack = curr_path + relative_ordatabase
os.environ['OPENRAVE_DATABASE'] = ordatabase_path_thispack

#get rid of warnings
openravepy.RaveInitialize(True, openravepy.DebugLevel.Fatal)
openravepy.misc.InitOpenRAVELogging()



class RoboHandler:
  def __init__(self):
    self.openrave_init()
    print ('Openrave init done ..')
    self.problem_init()
    print ('Problem init done ..')

    #order grasps based on your own scoring metric
    #self.order_grasps()
    #print ('Order grasps done ..')

    #order grasps with noise
    #self.order_grasps_noisy()
    #print ('Order grasps noisy done ..')


  # the usual initialization for openrave
  def openrave_init(self):
    self.env = openravepy.Environment()
    self.env.SetViewer('qtcoin')
    self.env.GetViewer().SetName('HW1 Viewer')
    self.env.Load('models/%s.env.xml' %PACKAGE_NAME)
    # time.sleep(3) # wait for viewer to initialize. May be helpful to uncomment
    self.robot = self.env.GetRobots()[0]
    self.manip = self.robot.GetActiveManipulator()
    self.end_effector = self.manip.GetEndEffector()

  # problem specific initialization - load target and grasp module
  def problem_init(self):
    self.target_kinbody = self.env.ReadKinBodyURI('models/objects/champagne.iv')
    #self.target_kinbody = self.env.ReadKinBodyURI('models/objects/winegoblet.iv')
    #self.target_kinbody = self.env.ReadKinBodyURI('models/objects/black_plastic_mug.iv')

    #change the location so it's not under the robot
    T = self.target_kinbody.GetTransform()
    T[0:3,3] += np.array([0.5, 0.5, 0.5])
    self.target_kinbody.SetTransform(T)
    self.env.AddKinBody(self.target_kinbody)

    # create a grasping module
    self.gmodel = openravepy.databases.grasping.GraspingModel(self.robot, self.target_kinbody)
    
    # if you want to set options, e.g. friction
    options = openravepy.options
    options.friction = 0.1
    if not self.gmodel.load():
      self.gmodel.autogenerate(options)

    self.graspindices = self.gmodel.graspindices
    self.grasps = self.gmodel.grasps

  
  # order the grasps - call eval grasp on each, set the 'performance' index, and sort
  def order_grasps(self):
    self.grasps_ordered = self.grasps.copy() #you should change the order of self.grasps_ordered
    i = 0
    for grasp in self.grasps_ordered:
      print('Evaluating grasp %d of %d' % (i, len(self.grasps_ordered)) )
      i, grasp[self.graspindices.get('performance')] = self.eval_grasp(grasp), i+1
    
    # sort!
    order = np.argsort(self.grasps_ordered[:,self.graspindices.get('performance')[0]])
    order = order[::-1]
    self.grasps_ordered = self.grasps_ordered[order]
  
  # order the grasps - but instead of evaluating the grasp, evaluate random perturbations of the grasp 
  def order_grasps_noisy(self):
    self.grasps_ordered_noisy = self.grasps.copy()

    # Order grasps based on std deviation in eval score due to uncertainty
    i, NUM_RAND_GRASPS, grasps_noisy_scores, grasps_raw_scores = 0, 5, [], []
    for grasp in self.grasps_ordered_noisy:
      print('Evaluating grasp %d of %d with uncertainty' % (i, len(self.grasps_ordered_noisy)) )
      grasps_raw_scores.append(self.eval_grasp(grasp))     
      rand_grasps =  [self.sample_random_grasp(grasp) for j in range(NUM_RAND_GRASPS)]
      rand_grasps_scores = [self.eval_grasp(rand_grasp) for rand_grasp in rand_grasps]    
      grasps_noisy_scores.append(1-np.std(rand_grasps_scores))
      i = i+1

    # Normalize the two metrics
    grasps_raw_scores_sum, grasps_noisy_scores_sum = sum(grasps_raw_scores), sum(grasps_noisy_scores)
    grasps_raw_scores = [score/grasps_raw_scores_sum for score in grasps_raw_scores]
    grasps_noisy_scores = [score/grasps_noisy_scores_sum for score in grasps_noisy_scores]

    # Update score with a weighted mean
    raw_scores_w, noisy_scores_w = 0.7, 0.3
    for i in range(len(self.grasps_ordered_noisy)):
      print('Compute final score %d of %d from individual scores %f, %f' % (i, len(self.grasps_ordered_noisy), grasps_raw_scores[i], grasps_noisy_scores[i]))
      self.grasps_ordered_noisy[i][self.graspindices.get('performance')] = [raw_scores_w*grasps_raw_scores[i] + noisy_scores_w*grasps_noisy_scores[i]]
     
    # sort!
    order = np.argsort(self.grasps_ordered_noisy[:,self.graspindices.get('performance')[0]])
    order = order[::-1]
    self.grasps_ordered_noisy = self.grasps_ordered_noisy[order]    

  # function to evaluate grasps
  # returns a score, which is some metric of the grasp
  # higher score should be a better grasp
  def eval_grasp(self, grasp):
    with self.robot:
      #contacts is a 2d array, where contacts[i,0-2] are the positions of contact i and contacts[i,3-5] is the direction
      try:
        contacts,finalconfig,mindist,volume = self.gmodel.testGrasp(grasp=grasp,translate=True,forceclosure=False)

        obj_position = self.gmodel.target.GetTransform()[0:3,3]
        # for each contact
        G = np.zeros((6, len(contacts))) #the wrench matrix
        for i, c in enumerate(contacts):
          pos = c[0:3] - obj_position
          dir = -c[3:] #this is already a unit vector
          
          torque = np.cross(pos, dir);

          G[:,i] = np.concatenate([dir, torque]);
          #TODO fill G
        
        [_, S, _] = np.linalg.svd(G)

        # Using minimization of singular values method
        singularValueMin = S[-1]

        # Maximization of the force ellipsoid volume method 
        #maxEllipsoidVol = math.sqrt(np.linalg.det(np.dot(G, G.transpose())));

        #return maxEllipsoidVol;
        return singularValueMin

      except:
        #you get here if there is a failure in planning
        #example: if the hand is already intersecting the object at the initial position/orientation
        return  0.00 # TODO you may want to change this

      
      #heres an interface in case you want to manipulate things more specifically
      #NOTE for this assignment, your solutions cannot make use of graspingnoise
#      self.robot.SetTransform(np.eye(4)) # have to reset transform in order to remove randomness
#      self.robot.SetDOFValues(grasp[self.graspindices.get('igrasppreshape')], self.manip.GetGripperIndices())
#      self.robot.SetActiveDOFs(self.manip.GetGripperIndices(), self.robot.DOFAffine.X + self.robot.DOFAffine.Y + self.robot.DOFAffine.Z)
#      self.gmodel.grasper = openravepy.interfaces.Grasper(self.robot, friction=self.gmodel.grasper.friction, avoidlinks=[], plannername=None)
#      contacts, finalconfig, mindist, volume = self.gmodel.grasper.Grasp( \
#            direction             = grasp[self.graspindices.get('igraspdir')], \
#            roll                  = grasp[self.graspindices.get('igrasproll')], \
#            position              = grasp[self.graspindices.get('igrasppos')], \
#            standoff              = grasp[self.graspindices.get('igraspstandoff')], \
#            manipulatordirection  = grasp[self.graspindices.get('imanipulatordirection')], \
#            target                = self.target_kinbody, \
#            graspingnoise         = 0.0, \
#            forceclosure          = True, \
#            execute               = False, \
#            outputfinal           = True, \
#            translationstepmult   = None, \
#            finestep              = None )



  # given grasp_in, create a new grasp which is altered randomly
  # you can see the current position and direction of the grasp by:
  # grasp[self.graspindices.get('igrasppos')]
  # grasp[self.graspindices.get('igraspdir')]
  def sample_random_grasp(self, grasp_in, dist_sigma=.01, angle_sigma=np.pi/24):
    grasp = grasp_in.copy()
    
    #sample random position
    RAND_DIST_SIGMA = dist_sigma #TODO you may want to change this
    pos_orig = grasp[self.graspindices['igrasppos']]

    grasp[self.graspindices['igrasppos']] = [np.random.normal(item, RAND_DIST_SIGMA, 1)[0] for item in grasp[self.graspindices['igrasppos']]]

    #sample random orientation
    RAND_ANGLE_SIGMA = angle_sigma#TODO you may want to change this
    dir_orig = grasp[self.graspindices['igraspdir']]
    roll_orig = grasp[self.graspindices['igrasproll']]
    grasp[self.graspindices['igraspdir']] = [np.random.normal(item, RAND_ANGLE_SIGMA, 1)[0] for item in grasp[self.graspindices['igraspdir']]]
    grasp[self.graspindices['igrasproll']] = [np.random.normal(item, RAND_ANGLE_SIGMA, 1)[0] for item in grasp[self.graspindices['igrasproll']]]

    return grasp


  #displays the grasp
  def show_grasp(self, grasp, delay=1.5):
    with openravepy.RobotStateSaver(self.gmodel.robot):
      with self.gmodel.GripperVisibility(self.gmodel.manip):
        time.sleep(0.1) # let viewer update?
        try:
          with self.env:
            contacts,finalconfig,mindist,volume = self.gmodel.testGrasp(grasp=grasp,translate=True,forceclosure=True)
            #if mindist == 0:
            #  print 'grasp is not in force closure!'
            contactgraph = self.gmodel.drawContacts(contacts) if len(contacts) > 0 else None
            self.gmodel.robot.GetController().Reset(0)
            self.gmodel.robot.SetDOFValues(finalconfig[0])
            self.gmodel.robot.SetTransform(finalconfig[1])
            self.env.UpdatePublishedBodies()
            time.sleep(delay)
        except openravepy.planning_error,e:
          print 'bad grasp!',e

if __name__ == '__main__':
  print ('Started')
  robo = RoboHandler()
  
  import IPython
  IPython.embed()
  while True:
    time.sleep(10000) #to keep the openrave window open

  
