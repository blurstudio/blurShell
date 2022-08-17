#pragma once

#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <maya/MPointArray.h>
#include <maya/MFloatPointArray.h>
#include <maya/MIntArray.h>
#include <maya/MVector.h>
#include <maya/MFloatVectorArray.h>

typedef std::pair<int, int> Edge;


struct PairHash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        auto h1 = std::hash<T1>()(pair.first);
        auto h2 = std::hash<T2>()(pair.second);

        // from boost::hash_combine
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};



/*
Naiively triangulate the given mesh
*/
MIntArray triangulateFaces(const MIntArray& faces, const MIntArray& counts){
    int triCount = 0;
    for (const auto &c: counts){
        triCount += c - 2;
    }

    MIntArray ret(triCount);
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
MIntArray reverseFaces(const MIntArray& faces, const MIntArray& counts, int offset) {
    MIntArray ret(faces.length());

    int idx = 0, pIdx = 0, j = 0;
    for (const auto &c: counts){
        idx += c;
        for (int i = idx - 1; i >= pIdx; --i){
            ret[j++] = faces[i] + offset;
        }
    }
    return ret;
}

/*
Insert sorted(a,b):(a,b) into the map if it's not already in there
If it IS in there, then remove it from the map
*/
void insertOrErase(std::unordered_map<Edge, Edge, PairHash> &map, int a, int b){
    Edge key, val;
    if (a < b){
        key = std::make_pair(a, b);
    }
    else {
        key = std::make_pair(b, a);
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
std::vector<Edge> findSortedBorderEdges(MIntArray& tris){
    // keep track of whether the edges have been seen an even
    // or odd number of times. Only border edges will be seen an odd
    // number of times.
    std::unordered_map<Edge, Edge, PairHash> map;
    for (uint i = 0; i < tris.length(); i += 3){
        insertOrErase(map, tris[i], tris[i + 2]);
        insertOrErase(map, tris[i + 1], tris[i]);
        insertOrErase(map, tris[i + 2], tris[i + 1]);
    }

    std::unordered_set<Edge, PairHash> borderSet;
    for (const auto& kv : map){
        borderSet.insert(kv.second);
    }

    std::vector<Edge> ret(borderSet.size());
    uint idx = 0;
    for (uint i = 0; i < tris.length(); i += 3){
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
    for (int i = 0; i < edges.size(); ++i){
        startMap[edges[i].first] = i;
    }
    std::vector<int> cycle;
    cycle.reserve(edges.size());
    for (size_t i = 0; i < edges.size(); ++i){
        cycle.push_back(startMap[edges[i].second]);
    }
    return cycle;
}

// Returns the border edge indices in order
MIntArray shellTopo(
    const MIntArray& oFaces, const MIntArray& ioCounts, int vertCount, int numBridgeLoops,
    MIntArray& faces, MIntArray& counts
) {

    counts.setLength(ioCounts.length() * 2);
    int ptr = 0;
    for (auto &c : ioCounts) {
        counts[ptr++] = c;
    }
    for (auto &c : ioCounts) {
        counts[ptr++] = c;
    }


    auto iFaces = reverseFaces(oFaces, ioCounts, vertCount);
    faces.setLength(oFaces.length() * 2);
    ptr = 0;
    for (auto &c : oFaces) {
        faces[ptr++] = c;
    }
    for (auto &c : iFaces) {
        faces[ptr++] = c;
    }

    // TODO: May not have to triangulate
    MIntArray tris = triangulateFaces(faces, counts);
    std::vector<Edge> obEdges = findSortedBorderEdges(tris);
    std::vector<int> cycle = buildCycles(obEdges);

    MIntArray firstEdges((uint)obEdges.size());
    ptr = 0;
    for (const auto &e: obEdges){
        firstEdges[ptr++] = e.first;
    }

    counts.setSizeIncrement((uint)((numBridgeLoops + 1) * obEdges.size()));
    faces.setSizeIncrement((uint)((numBridgeLoops + 1) * obEdges.size() * 4));

    for (int loopIdx = 0; loopIdx < numBridgeLoops; ++loopIdx){
        int curOffset = (loopIdx * (int)obEdges.size()) + vertCount;
        if (loopIdx == 0){
            // The first loop doesn't 
            curOffset -= vertCount;
        }
        int nxtOffset = ((loopIdx + 1) * (int)obEdges.size()) + vertCount;
        for (int i = 1; i < obEdges.size(); ++i){
            faces.append(obEdges[i].first + curOffset);
            faces.append(obEdges[i].first + nxtOffset);
            faces.append(obEdges[cycle[obEdges[i].first]].first + nxtOffset);
            faces.append(obEdges[cycle[obEdges[i].first]].first + curOffset);
            counts.append(4);
        }
    }

    int curOffset = 0;
    if (numBridgeLoops > 0){
        curOffset = (numBridgeLoops * (int)obEdges.size()) + vertCount;
    }
    int nxtOffset = vertCount;
    for (size_t i = 1; i < obEdges.size(); ++i){
        faces.append(obEdges[i].first + curOffset);
        faces.append(obEdges[i].first + nxtOffset);
        faces.append(obEdges[cycle[obEdges[i].first]].first + nxtOffset);
        faces.append(obEdges[cycle[obEdges[i].first]].first + curOffset);
        counts.append(4);
    }

    return firstEdges;
}


MFloatPointArray shellGeo(
    const MPointArray& rawVerts, const MFloatVectorArray& normals, const MIntArray& bIdxs,
    int numBridgeLoops, float innerOffset, float outerOffset
){

    MFloatPointArray ret;
    int initCount = rawVerts.length();

    ret.setLength((initCount * 2) + ((numBridgeLoops - 1) * bIdxs.length()));

    for (uint i = 0; i < rawVerts.length(); ++i){
        ret[i] = rawVerts[i] + (normals[i] * outerOffset);
        MVector x = normals[i] * innerOffset;
        ret[i + initCount] = rawVerts[i] - x;
    }

    int ptr = 2 * initCount;
    for (int i = 1; i < numBridgeLoops; ++i){
        float perc = (float)i / numBridgeLoops;
        for (const auto& bi: bIdxs){
            ret[ptr++] = ret[bi] * perc + ret[bi + initCount] * (1.0 - perc);
        }
    }

    return ret;
}




