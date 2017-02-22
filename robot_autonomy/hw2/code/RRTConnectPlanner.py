import numpy, operator
from RRTPlanner import RRTTree

class RRTConnectPlanner(object):

    def __init__(self, planning_env, visualize):
        self.planning_env = planning_env
        self.visualize = visualize
        
    def RandomGoalPlan(self, tree, goal_state):
        v_rand = self.planning_env.GenerateRandomConfiguration(goal_state)
        v_near_id, v_near = tree.GetNearestVertex(v_rand)
        v_new = self.planning_env.Extend(v_near, v_rand)
        if v_new is not None:
            v_new_id = tree.AddVertex(v_new)
            tree.AddEdge(v_near_id, v_new_id)
            if self.planning_env.visualize:
                self.planning_env.PlotEdge(v_near, v_new)
        return v_new

    def ConnectPlan(self, tree, target, epsilon):
        v_near_id, v_near = tree.GetNearestVertex(target)
        v_new = self.planning_env.Extend(v_near, target)

        # Try connecting to target by multiple extend steps
        while (v_new is not None):
            v_new_id = tree.AddVertex(v_new)
            tree.AddEdge(v_near_id, v_new_id)    
            if self.planning_env.visualize:
                self.planning_env.PlotEdge(v_near, v_new)
            if self.planning_env.ComputeDistance(v_new, target) < epsilon:
                target_id = tree.AddVertex(target)
                tree.AddEdge(v_new_id, target_id)    
                if self.planning_env.visualize:
                    self.planning_env.PlotEdge(v_new, target)
                return True
            else:
                v_near_id, v_near = v_new_id, v_new            
                v_new = self.planning_env.Extend(v_near, target)
        return False
        

    def Plan(self, start_config, goal_config, epsilon = 0.2):
        ftree = RRTTree(self.planning_env, start_config)
        rtree = RRTTree(self.planning_env, goal_config)
        plan = []

        if self.visualize and hasattr(self.planning_env, 'InitializePlot'):
            self.planning_env.InitializePlot(goal_config)

        NUM_ITERATIONS = 100000
        for iter in range(NUM_ITERATIONS):

            # pick random goal tree and connect path tree
            rand_tree, rand_goal_config = (ftree, goal_config) if len(ftree.vertices) < len(rtree.vertices) else (rtree, start_config)
            connect_tree = ftree if len(ftree.vertices) >= len(rtree.vertices) else rtree

            # random goal plan
            v_new = self.RandomGoalPlan(rand_tree, rand_goal_config)
            # connect path plan
            if (v_new is not None) and self.ConnectPlan(connect_tree, v_new, epsilon):
                break     

            if (iter%10 == 0): 
                d_f = self.planning_env.ComputeDistance(ftree.GetNearestVertex(goal_config)[1], goal_config)
                d_r = self.planning_env.ComputeDistance(rtree.GetNearestVertex(start_config)[1], start_config)
                print(iter, ' Closest ftree dist to end goal :', d_f, '; Closest rtree dist to start goal :', d_r)

        dist = self.planning_env.ComputeDistance(ftree.vertices[-1], rtree.vertices[-1]) 
        print("Dist=", dist)
        if dist < epsilon:
            plan_f, fid = [ftree.vertices[-1]], len(ftree.vertices)-1
            while (fid != ftree.GetRootId()):
                plan_f.append(ftree.vertices[ftree.edges[fid]])
                fid = ftree.edges[fid]                 
            plan_r, rid = [rtree.vertices[-1]], len(rtree.vertices)-1
            while (rid != rtree.GetRootId()):
                plan_r.append(rtree.vertices[rtree.edges[rid]])
                rid = rtree.edges[rid]                 
            plan = plan_f[::-1]
            plan.extend(plan_r)

            if self.visualize and hasattr(self.planning_env, 'InitializePlot'):
                self.planning_env.InitializePlot(goal_config)
                if self.planning_env.visualize:
                    [self.planning_env.PlotEdge(plan[i-1], plan[i]) for i in range(1,len(plan))]
                          
        return plan
