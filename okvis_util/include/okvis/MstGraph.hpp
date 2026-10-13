/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file MstGraph.hpp
 * @brief Maximum Spanning Tree Computation.
 * @author Stefan Leutenegger
 */

#ifndef MSTGRAPH_HPP
#define MSTGRAPH_HPP

#include <vector>

namespace okvis {

struct MstGraph
{
    int V; 
    int E; 
    std::vector< std::pair<int, std::pair<int, int>> > edges; 

    struct DisjointSets
    {
        int *parent; 
        int *rnk; 
        int n; 

        DisjointSets(int n)
        {

            this->n = n;
            parent = new int[size_t(n+1)];
            rnk = new int[size_t(n+1)];

            for (int i = 0; i <= n; i++)
            {
                rnk[i] = 0;

                parent[i] = i;
            }
        }

        ~DisjointSets() {
            delete[] parent;
            delete[] rnk;
        }

        int find(int u)
        {

            if (u != parent[u])
                parent[u] = find(parent[u]);
            return parent[u];
        }

        void merge(int x, int y)
        {
            x = find(x);
            y = find(y);

            if (rnk[x] > rnk[y])
                parent[y] = x;
            else 
                parent[x] = y;

            if (rnk[x] == rnk[y])
                rnk[y]++;
        }
    };

    MstGraph(int V, int E)
    {
        this->V = V;
        this->E = E;
    }

    void addEdge(int u, int v, int w)
    {
        edges.push_back({w, {u, v}});
    }

    int kruskalMst(std::vector<std::pair<int, int> > &edgesOut);
};

int MstGraph::kruskalMst(std::vector<std::pair<int,int>> & edgesOut)
{
    int mst_wt = 0; 

    std::sort(edges.begin(), edges.end());

    MstGraph::DisjointSets ds(V);

    std::vector< std::pair<int, std::pair<int, int> > >::iterator it;
    for (it=edges.begin(); it!=edges.end(); it++)
    {
        int u = it->second.first;
        int v = it->second.second;

        int set_u = ds.find(u);
        int set_v = ds.find(v);

        if (set_u != set_v)
        {

            edgesOut.push_back({u,v});

            mst_wt += it->first;

            ds.merge(set_u, set_v);
        }
    }

    return mst_wt;
}

} 

#endif 
