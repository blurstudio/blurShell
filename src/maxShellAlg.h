#pragma once

#include <vector>
#include <utility>
#include <unordered_map>
#include <algorithm>
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

    MIntArray ret(triCount * 3);
    int start = 0, end = 0;
    int ptr = 0;
    for (const auto &c: counts){
        end += c;
        // Gotta triangulate from the *end* of the fan
        // to make the numbers work out
        for (int i = end - 2; i > start; i--){
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
        pIdx = idx;
    }
    return ret;
}


/*
Insert sorted(a,b):(a,b) into the map if it's not already in there
If it IS in there, then remove it from the map
*/
void insertOrErase(std::unordered_map<Edge, std::pair<Edge, uint>, PairHash> &map, int a, int b, uint idx){
    Edge key, val;
    if (a < b){
        key = std::make_pair(a, b);
    }
    else {
        key = std::make_pair(b, a);
    }
    if (map.find(key) == map.end()){
        map[key] = std::make_pair(std::make_pair(a, b), idx);
    }
    else{
        map.erase(key);
    }
}


bool eiSort(const std::pair<Edge, uint>& a, const std::pair<Edge, uint>& b) {
    return a.second < b.second;
}


/*
Build a vector of border edges in the order they appear in the given triangles
*/
std::vector<Edge> findSortedBorderEdges(const MIntArray& oFaces, const MIntArray& ioCounts){
    // TODO: May not have to triangulate
    MIntArray tris = triangulateFaces(oFaces, ioCounts);

    // Find the border edges and keep track of their index in the search
    // keep track of whether the edges have been seen an even
    // or odd number of times. Only border edges will be seen an odd
    // number of times.
    std::unordered_map<Edge, std::pair<Edge, uint>, PairHash> map;
    for (uint i = 0; i < tris.length(); i += 3){
        // These indices MUST be in this exact order
        insertOrErase(map, tris[i + 1], tris[i + 0], i + 0);
        insertOrErase(map, tris[i + 2], tris[i + 1], i + 1);
        insertOrErase(map, tris[i + 0], tris[i + 2], i + 2);
    }

    // Put the values of the map into a vector
    // Then sort that vector by the edge index
    std::vector<std::pair<Edge, uint>> idxPairs;
    idxPairs.reserve(map.size());
    for (const auto& kv : map) {
        idxPairs.push_back(kv.second);
    }
    std::sort(idxPairs.begin(), idxPairs.end(), eiSort);

    // Pull the pairs out into their own vector, and return
    std::vector<Edge> ret(map.size());
    for (size_t i = 0; i < map.size(); ++i) {
        ret[i] = idxPairs[i].first;
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


 int getBridgeIdx(uint eIdx, int segIdx, int numBridgeSegs, int vertCount, const MIntArray& bIdxs) {
     if (segIdx == 0) {
         return bIdxs[eIdx];
     }
     if (segIdx == numBridgeSegs) {
         return bIdxs[eIdx] + vertCount;
     }
     return (2 * vertCount) + eIdx + (bIdxs.length() * (segIdx - 1));
 }


// Returns the border edge indices in order
MIntArray shellTopo(
    const MIntArray& oFaces, const MIntArray& ioCounts, int vertCount, int numBridgeSegs,
    MIntArray& faces, MIntArray& counts
) {
    std::vector<Edge> obEdges = findSortedBorderEdges(oFaces, ioCounts);
    std::vector<int> cycle = buildCycles(obEdges);

    MIntArray firstEdges((uint)obEdges.size());
    int eptr = 0;
    for (const auto &e : obEdges) {
        firstEdges[eptr++] = e.first;
    }

    counts.setLength((ioCounts.length() * 2) + (numBridgeSegs * (uint)firstEdges.length()));
    int cptr = 0;
    for (auto &c : ioCounts) { counts[cptr++] = c; }
    for (auto &c : ioCounts) { counts[cptr++] = c; }

    auto iFaces = reverseFaces(oFaces, ioCounts, vertCount);
    faces.setLength((oFaces.length() * 2) + (numBridgeSegs * (uint)firstEdges.length() * 4));
    int fptr = 0;
    for (auto &c : oFaces) { faces[fptr++] = c; }
    for (auto &c : iFaces) { faces[fptr++] = c; }

    int inOffset = 0;
    for (int segIdx = 0; segIdx < numBridgeSegs; segIdx++) {
        for (uint eIdx = 0; eIdx < firstEdges.length(); ++eIdx) {
            faces[fptr++] = getBridgeIdx(eIdx       , segIdx    , numBridgeSegs, vertCount, firstEdges);
            faces[fptr++] = getBridgeIdx(eIdx       , segIdx + 1, numBridgeSegs, vertCount, firstEdges);
            faces[fptr++] = getBridgeIdx(cycle[eIdx], segIdx + 1, numBridgeSegs, vertCount, firstEdges);
            faces[fptr++] = getBridgeIdx(cycle[eIdx], segIdx    , numBridgeSegs, vertCount, firstEdges);
            counts[cptr++] = 4;
        }
    }

    return firstEdges;
}


MFloatPointArray shellGeo(
    const MPointArray& rawVerts, const MFloatVectorArray& normals, const MIntArray& bIdxs,
    int numBridgeSegs, float innerOffset, float outerOffset
){
    MFloatPointArray ret;
    int initCount = rawVerts.length();

    ret.setLength((initCount * 2) + ((numBridgeSegs - 1) * bIdxs.length()));

    for (uint i = 0; i < rawVerts.length(); ++i){
        ret[i] = rawVerts[i] + (normals[i] * outerOffset);
        ret[i + initCount] = rawVerts[i] - (MVector)(normals[i] * innerOffset);
    }

    for (uint eIdx = 0; eIdx < bIdxs.length(); ++eIdx) {
        auto innerIdx = getBridgeIdx(eIdx, 0            , numBridgeSegs, initCount, bIdxs);
        auto outerIdx = getBridgeIdx(eIdx, numBridgeSegs, numBridgeSegs, initCount, bIdxs);
        for (int segIdx = 1; segIdx < numBridgeSegs; segIdx++) {
            float perc = (float)segIdx / numBridgeSegs;
            auto curIdx = getBridgeIdx(eIdx, segIdx, numBridgeSegs, initCount, bIdxs);
            ret[curIdx] = ret[innerIdx] * (1.0 - perc) + ret[outerIdx] * perc;
        }
    }
    return ret;
}
