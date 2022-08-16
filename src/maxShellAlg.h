#pragma once

#include <vector>
#include <pair>
#include <unordered_map>
#include <unordered_set>
#include <maya/MPointArray.h>
#include <maya/MIntArray.h>
#include <maya/MFloatVectorArray.h>

typedef std::pair<int, int> Edge;



/*
Naiively triangulate the given mesh
*/
std::vector<int> triangulateFaces(const std::vector<int>& faces, const std::vector<int>& counts){
    int triCount = 0;
    for (const auto &c: counts){
        triCount += c - 2;
    }

    std::vector<int> ret;
    ret.setLength(triCount);

    int start = 0, end = 0;
    int ptr = 0;
    for (const auto &c: counts){
        end += c;
        for (int i = start + 1; i < end; i += 2){
            ret[ptr++] = faces[start];
            ret[ptr++] = faces[i];
            ret[ptr++] = faces[i + 1];
        }
        start = end;
    }
    return ret;   
}

/*
Given a mesh, reverse the order of each face so its normal goes in the
opposite direction
*/
template <class T>
std::vector<int> reverseFaces(const T& faces, const T& counts, int offset) {
    std::vector<int> ret;

    int idx = 0, pIdx = 0, j = 0;
    for (const auto &c: counts){
        idx += c;
        for (int i = idx - 1; i >= pIdx; --i){
            ret.push_back(faces[i] + offset);
        }
    }
    return ret;
}

/*
Insert sorted(a,b):(a,b) into the map if it's not already in there
If it IS in there, then remove it from the map
*/
void insertOrErase(EdgeMap &map, int a, int b){
    Edge key, val;
    if (a < b){
        key = std::make_pair(a, b);
    }
    else {
        key = std::make_pair(b, a)
    }
    if (map.find(key) != map.end()){
        map[key] = std::make_pair(a, b);
    }
    else{
        map.erase(key);
    }
}

/*
Build a vector of border edges in the order they appear in the given triangles
*/
std::vector<Edge> findSortedBorderEdges(const std::vector<int>& tris){
    // keep track of whether the edges have been seen an even
    // or odd number of times. Only border edges will be seen an odd
    // number of times.
    std::unordered_map<Edge, Edge> map;
    for (int i = 0; i < tris.size(); i += 3){
        insertOrErase(map, tris[i], tris[i + 2]);
        insertOrErase(map, tris[i + 1], tris[i]);
        insertOrErase(map, tris[i + 2], tris[i + 1]);
    }

    std::unordered_set<Edge> borderSet;
    for (std::pair<Edge, Edge> &kv : map){
        borderSet.insert(kv.second);
    }

    std::vector<Edge> ret(borderSet.size());
    int idx = 0;
    for (int i = 0; i < tris.size(); i += 3){
        if (borderSet.find(std::make_pair(tris[i], tris[i + 1])) != borderSet.end()){
            ret[idx++] = std::make_pair(tris[i], tris[i + 1]);
        }
        if (borderSet.find(std::make_pair(tris[i + 1], tris[i + 2])) != borderSet.end()){
            ret[idx++] = std::make_pair(tris[i + 1], tris[i + 2]);
        }
        if (borderSet.find(std::make_pair(tris[i + 2], tris[i])) != borderSet.end()){
            ret[idx++] = std::make_pair(tris[i + 2], tris[i]);
        }
    }
    return ret;
}

std::vector<int> buildCycles(const std::vector<Edge>& edges){
    std::unordered_map<int, int> startMap;
    for (size_t i = 0; i < edges.size(); ++i){
        startMap[edges[i].first] = i;
    }
    std::vector<int> cycle;
    cycle.reserve(edges.size());
    for (size_t i = 0; i < edges.size(); ++i){
        cycle.push_back(startMap[edges[i].second]);
    }
    return cycle
}

// Returns the border edge indices in order
std::vector<int> shellTopo(
    const MIntArray& oFaces, const MIntArray& ioCounts, int vertCount, int numBridgeLoops,
    std::vector<int>& faces, std::vector<int>& counts, 
) {
    // The outer faces stay the same.  The inner faces get reversed
    auto iFaces = reverseFaces(oFaces, ioCounts, vertCount);

    counts.insert(ioCounts.begin(), ioCounts.end());
    counts.insert(counts.end(), ioCounts.begin(), ioCounts.end());

    faces.insert(oFaces.begin(), oFaces.end());
    faces.insert(faces.end(), iFaces.begin(), iFaces.end());

    // TODO: May not have to triangulate
    std::vector<int> tris = triangulateFaces(faces, counts);
    std::vector<Edge> obEdges = findSortedBorderEdges(tris);
    std::vector<int> cycle = buildCycles(obEdges);

    std::vector<int> firstEdges;
    firstEdges.reserve(obEdges.size());
    for (const auto &e: obEdges){
        firstEdges.push_back(e.first);
    }

    counts.reserve(counts.size() + ((numBridgeLoops + 1) * obEdges.size()) );
    faces.reserve(faces.size() + ((numBridgeLoops + 1) * obEdges.size() * 4));

    for (size_t loopIdx = 0; loopIdx < numBridgeLoops; ++loopIdx){
        int curOffset = (loopIdx * obEdges.size()) + vertCount;
        if (loopIdx == 0){
            // The first loop doesn't 
            curOffset -= vertCount;
        }
        int nxtOffset = ((loopIdx + 1) * obEdges.size()) + vertCount;
        for (size_t i = 1; i < obEdges.size(); ++i){
            faces.push_back(obEdges[i].first + curOffset);
            faces.push_back(obEdges[i].first + nxtOffset);
            faces.push_back(obEdges[cycle[obEdges[i].first]].first + nxtOffset);
            faces.push_back(obEdges[cycle[obEdges[i].first]].first + curOffset);
            counts.push_back(4);
        }
    }

    int curOffset = 0;
    if (numBridgeLoops > 0){
        curOffset = (numBridgeLoops * obEdges.size()) + vertCount;
    }
    int nxtOffset = vertCount;
    for (size_t i = 1; i < obEdges.size(); ++i){
        faces.push_back(obEdges[i].first + curOffset);
        faces.push_back(obEdges[i].first + nxtOffset);
        faces.push_back(obEdges[cycle[obEdges[i].first]].first + nxtOffset);
        faces.push_back(obEdges[cycle[obEdges[i].first]].first + curOffset);
        counts.push_back(4);
    }

    return firstEdges;
}


MPointArray shellGeo(
    const MPointArray& rawVerts, const MFloatVectorArray& normals, const std::vector<int>& bIdxs,
    int numBridgeLoops, float innerOffset, float outerOffset
){

    MPointArray ret;
    int initCount = rawVerts.length();

    ret.setLength((initCount * 2) + ((numBridgeLoops - 1) * bIdxs.length()));

    for (int i = 0; i < rawVerts.length(); ++i){
        ret[i] = rawVerts[i] + (normals[i] * outerOffset);
        ret[i + initCount] = rawVerts[i] - (normals[i] * innerOffset);
    }

    int ptr = 2 * initCount;
    for (int i = 1; i < numBridgeLoops; ++i){
        float perc = i / numBridgeLoops;
        for (const auto& bi: bIdxs){
            ret[ptr++] = ret[bi] * perc + ret[bi + initCount] * (1.0 - perc);
        }
    }

    return ret;
}




