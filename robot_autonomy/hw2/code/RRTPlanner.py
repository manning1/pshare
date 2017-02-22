import numpy
from RRTTree import RRTTree

class RRTPlanner(object):

    def __init__(self, planning_env, visualize):
        self.planning_env = planning_env
        self.visualize = visualize

    def Plan(self, start_config, goal_config, epsilon = 0.2):
      
        print("Epsilon=", epsilon)  
        tree = RRTTree(self.planning_env, start_config)
        plan = []
        if self.visualize and hasattr(self.planning_env, 'InitializePlot'):
            self.planning_env.InitializePlot(goal_config)

        self.planning_env.SetGoalParameters(goal_config)        

        NUM_ITERATIONS = 100000
        for iter in range(NUM_ITERATIONS):

            # Check if close enough to goal
            vid, v = tree.GetNearestVertex(goal_config)
            d = self.planning_env.ComputeDistance(v, goal_config)
            if (iter%10 == 0): 
                print(iter, ' Closest dist to goal :', d)

            if ((d is not None) and  (d < epsilon)):
                tree.AddEdge(vid, tree.AddVertex(goal_config))
                if self.planning_env.visualize:
                    self.planning_env.PlotEdge(v, goal_config)
                print ("Goal Found !!!")
                break

            v_rand = self.planning_env.GenerateRandomConfiguration()
            v_near_id, v_near = tree.GetNearestVertex(v_rand)
            v_new = self.planning_env.Extend(v_near, v_rand)
            if v_new is not None:
                v_new_id = tree.AddVertex(v_new)
                tree.AddEdge(v_near_id, v_new_id)
                if self.planning_env.visualize:
                    self.planning_env.PlotEdge(v_near, v_new)
    
        # If goal found, evaluate path
        if self.planning_env.ComputeDistance(tree.vertices[-1], goal_config) == 0:            
            plan.append(goal_config)
            id = len(tree.vertices)-1
            while (id != tree.GetRootId()):
                plan.append(tree.vertices[tree.edges[id]])
                id = tree.edges[id]                 
        
        plan = plan[::-1]
        if self.visualize and hasattr(self.planning_env, 'InitializePlot'):
            self.planning_env.InitializePlot(goal_config)
            if self.planning_env.visualize:
                [self.planning_env.PlotEdge(plan[i-1], plan[i]) for i in range(1,len(plan))]

        return plan
