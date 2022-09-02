#pragma once

#include <vector>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include <cmath>


typedef std::pair<int, int> Edge;
// Use a structure of arrays because I have to 
typedef std::pair<std::vector<int>, std::vector<int>> EdgeVector;


/*
A hash function for std::pairs taken from boost::hash_combine
*/
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
Given a mesh, reverse the order of each face so its normal goes in the
opposite direction
*/
std::vector<int> reverseFaces(const std::vector<int>& faces, const std::vector<int>& counts, int offset) {
    std::vector<int> ret(faces.size());

    int fIdx = 0, rIdx = 0;
    for (const auto &c: counts){
        ret[rIdx] = faces[fIdx++] + offset;
        for (int i = 1; i < c; ++i){
            ret[rIdx + c - i] = faces[fIdx++] + offset;
        }
        rIdx += c;
    }
    return ret;
}

/*
Insert a mapping from (sorted edge pair):(ordered edge pair) if the sorted
edge pair isn't already in there.
If it IS already in there, then remove it from the map
This ensures that only edges with an odd number of connected faces
(ie, border edges) will be in the map once we've looped through all the edges
*/
void insertOrErase(std::unordered_map<Edge, int, PairHash> &map, int a, int b, int idx){
    Edge key, val;
    if (a < b){
        key = std::make_pair(a, b);
    }
    else {
        key = std::make_pair(b, a);
    }
    if (map.find(key) == map.end()){
        map[key] = idx;
    }
    else{
        map.erase(key);
    }
}


/*
A quick function to sort edge pairs by their second index
*/
bool eiSort(const std::pair<Edge, int>& a, const std::pair<Edge, int>& b) {
    return a.second < b.second;
}

/*
Get the index of the next edge around each face
*/
std::vector<int> getEdgePairIdxs(const std::vector<int>& counts, const std::vector<int>& faces) {
    std::vector<int> ret(faces.size());
    int fIdx = 0, rIdx = 0;
    for (const auto &c: counts){
        for (int i = 1; i < c; ++i){
            ret[rIdx++] = faces[fIdx + i];
        }
        ret[rIdx++] = faces[fIdx];
        fIdx += c;
    }
    return ret;
}

/*
Build a vector of border face-vert edges in the order they appear in the given faces
*/
EdgeVector findFaceVertBorderPairs(const std::vector<int>& counts, const std::vector<int>& faces) {
    std::vector<int> nextIdxs = getEdgePairIdxs(counts, faces);
    std::unordered_map<Edge, int, PairHash> edgeDict;

    for (int eIdx = 0; eIdx < nextIdxs.size(); ++eIdx) {
        insertOrErase(edgeDict, faces[eIdx], faces[nextIdxs[eIdx]], eIdx);
    }

    std::vector<int> startIdxs, endIdxs;
    startIdxs.reserve(edgeDict.size());
    endIdxs.reserve(edgeDict.size());
    for (const auto &it : edgeDict){
        startIdxs.push_back(it.second);
    }
    std::sort(startIdxs.begin(), startIdxs.end());
    for (const auto &v: startIdxs){
        endIdxs.push_back(nextIdxs[v]);
    }

    EdgeVector fvibp;  //face vert index border pairs
    fvibp.first.reserve(endIdxs.size());
    fvibp.second.reserve(endIdxs.size());
    for (int i = 0; i < endIdxs.size(); ++i){
        fvibp.first.push_back(startIdxs[i]);
        fvibp.second.push_back(endIdxs[i]);
    }
    return fvibp;
}

/*
Find the indices where each needle exists in the haystack
If the needle isn't in there, the value will be -1
*/
template <typename T>
std::vector<T> findInArray(const std::vector<T>& needles, const std::vector<T>& haystack){
    std::unordered_map<T, T> hayMap;
    for (T i = 0; i < haystack.size(); ++i){
        hayMap[haystack[i]] = i;
    }

    std::vector<T> ret(needles.size());
    
    for (T i = 0; i < needles.size(); ++i){
        auto hiter = haystack.find(i);
        if (hiter == haystack.end()){
            needles[i] = -1;
        }
        else{
            needles[i] = hiter->second;
        }
    }
    return ret;
}


/*
Return the border verts and edge pairs in 3dsMax order
*/
std::tuple<std::vector<int>, EdgeVector>
findSortedBorderEdges(
    const std::vector<int>& oFaces,
    const std::vector<int>& ioCounts,
    const EdgeVector& faceVertIdxBorderPairs
){
    EdgeVector edges;
    edges.first.reserve(faceVertIdxBorderPairs.first.size());
    edges.second.reserve(faceVertIdxBorderPairs.first.size());

    EdgeVector sortedEdges(edges);
    std::sort(sortedEdges.first.begin(), sortedEdges.first.end());
    std::sort(sortedEdges.second.begin(), sortedEdges.second.end());

    std::vector<int> stopVals;
    std::set_difference(
        sortedEdges.first.begin(), sortedEdges.first.end(),
        sortedEdges.second.begin(), sortedEdges.second.end(),
        std::back_inserter(stopVals)
    );

    if (stopVals.empty()){
        return {edges.first, edges};
    }

    std::vector<int> stopIdxs = findInArray(stopVals, edges.second);

    std::vector<int> bVerts;
    bVerts.reserve(edges.first.size() + stopIdxs.size());
    for (int p = 0, i = 0; i < edges.first.size(); ++i){
        if (stopIdxs[p] == i){
            bVerts.push_back(edges.second[i]);
            ++ p;
        }
        bVerts.push_back(edges.first[i]);
    }
    return {bVerts, edges};
}


std::tuple<std::vector<int>, std::vector<int>, std::vector<std::vector<int>>>
buildBridgesByEdge(
    const std::vector<int>& bVerts,
    const EdgeVector& edges,
    int bridgeFirstIdx,
    int numSegs
){
    EdgeVector edgeInsertIdxs;
    edgeInsertIdxs.second = findInArray(edges.first, bVerts);
    edgeInsertIdxs.first = findInArray(edges.second, bVerts);

    int gridHeight = numSegs + 1;
    std::vector<std::vector<int>> grid(gridHeight, std::vector<int>(bVerts.size(), -1));
    grid[0] = bVerts;

    int ptr = bridgeFirstIdx;
    for (int i = 0; i < edgeInsertIdxs.first.size(); ++i){
        int eStart = edgeInsertIdxs.first[i];
        int eEnd = edgeInsertIdxs.second[i];
        for (int x = 0; x < gridHeight; ++x){
            int setter = grid[x][eStart];
            if (setter == -1)
                grid[x][eStart] = ptr++;
            setter = grid[x][eEnd];
            if (setter == -1)
                grid[x][eEnd] = ptr++;
        }
    }

    std::vector<int> faces;
    faces.reserve(numSegs * (bVerts.size() - 1));
    for (int i = 0; i < grid.size() - 1; ++i){
        for (int j = 0; j < grid[i].size() - 1; ++j){
            faces.push_back(grid[i + 1][edgeInsertIdxs.second[j]]);
            faces.push_back(grid[i + 1][edgeInsertIdxs.first[j]]);
            faces.push_back(grid[i][edgeInsertIdxs.first[j]]);
            faces.push_back(grid[i][edgeInsertIdxs.second[j]]);
        }
    }

    std::vector<int> counts((edges.first.size() * numSegs), 4);
    return {faces, counts, grid};
}


std::tuple<std::vector<int>, std::vector<int>, std::vector<std::vector<int>>>
buildBridgesByVert(
    const std::vector<int>& bVerts,
    const std::vector<int>& eVerts,
    const EdgeVector& edges,
    int bridgeFirstIdx,
    int numSegs
){
    int gridHeight = numSegs + 1;
    std::vector<std::vector<int>> grid(gridHeight, std::vector<int>(bVerts.size(), -1));
    grid[0] = bVerts;
    grid[numSegs] = eVerts;

    for (int i = 0; i < numSegs; ++i){
        for (int j = 0; j < bVerts.size(); ++j){
            grid[i][j] = j + bridgeFirstIdx + ((i - 1) * bVerts.size());
        }
    }
    EdgeVector edgeInsertIdxs;
    edgeInsertIdxs.second = findInArray(edges.first, bVerts);
    edgeInsertIdxs.first = findInArray(edges.second, bVerts);

    std::vector<int> faces;
    faces.reserve(numSegs * (bVerts.size() - 1));
    for (int i = 0; i < grid.size() - 1; ++i){
        for (int j = 0; j < grid[i].size() - 1; ++j){
            faces.push_back(grid[i + 1][edgeInsertIdxs.second[j]]);
            faces.push_back(grid[i + 1][edgeInsertIdxs.first[j]]);
            faces.push_back(grid[i][edgeInsertIdxs.first[j]]);
            faces.push_back(grid[i][edgeInsertIdxs.second[j]]);
        }
    }

    std::vector<int> counts((edges.first.size() * numSegs), 4);
    return {faces, counts, grid};
}



std::tuple<std::vector<int>, std::vector<int>, std::vector<std::vector<int>>, EdgeVector, std::vector<int>>
shellUvTopo(
    const EdgeVector& faceVertIdxBorderPairs,
    const std::vector<int>& oUvFaces,
    const std::vector<int>& oUvCounts,
    int numUVs,
    int numBridgeSegs
){
    auto [bVerts, edges] = findSortedBorderEdges(oUvCounts, oUvFaces, faceVertIdxBorderPairs);

    int bridgeFirstIdx = numUVs * 2;
    auto [bFaces, bCounts, grid] = buildBridgesByEdge(
        bVerts, edges, bridgeFirstIdx, numBridgeSegs
    );

    std::vector<int> iUvFaces = reverseFaces(oUvCounts, oUvFaces, 0);

    std::vector<int> faces(oUvFaces);
    faces.insert(faces.end(), iUvFaces.rbegin(), iUvFaces.rend());
    faces.insert(faces.end(), bFaces.begin(), bFaces.end());

    std::vector<int> counts(oUvCounts);
    counts.insert(counts.end(), oUvCounts.begin(), oUvCounts.end());
    counts.insert(counts.end(), bCounts.begin(), bCounts.end());

    return {faces, counts, grid, edges, bVerts};
}


std::tuple<std::vector<int>, std::vector<int>, std::vector<std::vector<int>>, std::vector<int>>
shellTopo(
    const EdgeVector& faceVertIdxBorderPairs,
    const std::vector<int>& oFaces,
    const std::vector<int>& oCounts,
    int vertCount,
    int numBridgeSegs
){
    auto [bVerts, edges] = findSortedBorderEdges(oCounts, oFaces, faceVertIdxBorderPairs);

    std::vector<int> eVerts;
    eVerts.reserve(bVerts.size());
    for (int i = 0; i < bVerts.size(); ++i){
        eVerts.push_back(bVerts[i] + vertCount);
    }
    int bridgeFirstIdx = 2 * vertCount;
    
    auto [bFaces, bCounts, grid] = buildBridgesByVert(
        bVerts, eVerts, edges, bridgeFirstIdx, numBridgeSegs
    );

    std::vector<int> iFaces = reverseFaces(oCounts, oFaces, 0);

    std::vector<int> faces(oFaces);
    faces.insert(faces.end(), iFaces.rbegin(), iFaces.rend());
    faces.insert(faces.end(), bFaces.begin(), bFaces.end());
    
    std::vector<int> counts(oCounts);
    counts.insert(counts.end(), oCounts.begin(), oCounts.end());
    counts.insert(counts.end(), bCounts.begin(), bCounts.end());

    return {counts, faces, grid, bVerts};
}


std::vector<float> _getOffsettedUvs(
    const std::vector<float>& uvs,
    const std::vector<std::vector<int>>& grid,
    const EdgeVector& edges,
    float offset
){
    const auto& bVerts = grid[0];
    
    auto nxtFind = findInArray(bVerts, edges.second);
    std::vector<bool> midPtBool(nxtFind.size(), true);
    std::vector<int> nxtIdxs, noNxts;
    nxtIdxs.reserve(nxtFind.size());
    for (int i = 0; i < nxtFind.size(); ++i){
        int f = edges.first[nxtFind[i]];
        if (f == -1){
            noNxts.push_back(nxtIdxs.size());
            nxtIdxs.push_back(-1);
            midPtBool[i] = false;
        }
        else {
            nxtIdxs.push_back(edges.first[f]);
        }
    }
    
    auto prevFind = findInArray(bVerts, edges.first);
    std::vector<int> prevIdxs, noPrevs;
    prevIdxs.reserve(prevFind.size());
    for (int i = 0; i < prevFind.size(); ++i){
        int f = edges.second[prevFind[i]];
        if (f == -1){
            noPrevs.push_back(prevIdxs.size());
            prevIdxs.push_back(-1);
            midPtBool[i] = false;
        }
        else {
            prevIdxs.push_back(edges.second[f]);
        }
    }

    std::vector<int> midPts;
    for (int i = 0; i < midPtBool.size(); ++i){
        if (midPtBool[i]){
            midPts.push_back(i);
        }
    }

    std::vector<float> outerVerts;
    for (int i = 0; i < bVerts.size(); ++i){
        int pI, nI, bI;
        float pU, pV, nU, nV, bU, bV;
        pI = prevIdxs[i];
        nI = nxtIdxs[i];
        bI = bVerts[i];

        bU = uvs[bI * 2];
        bV = uvs[bI * 2 + 1];

        if (pI != -1){
            pU = uvs[pI * 2] - bU;
            pV = uvs[pI * 2 + 1] - bV;
            float pLen = std::sqrt(pU * pU + pV * pV);
            pU /= pLen;
            pV /= pLen;
        }
        if (nI != -1){
            nU = uvs[nI * 2] - bU;
            nV = uvs[nI * 2 + 1] - bV;
            float nLen = std::sqrt(nU * nU + nV * nV);
            nU /= nLen;
            nV /= nLen;
        }

        if (pI == -1){
            outerVerts.push_back(pV * offset + bU);
            outerVerts.push_back(-pU * offset + bV);
            continue;
        }
        if (nI == -1){
            outerVerts.push_back(-nV * offset + bU);
            outerVerts.push_back(nU * offset + bV);
            continue;
        }

        // s for subtract
        float sx = pU - nU;
        float sy = pV - nV;
        float sl = std::sqrt(sx * sx + sy * sy);

        // a for add
        float ax = pU + nU;
        float ay = pV + nV;
        float al = std::sqrt(ax * ax + ay * ay);

        float halfAngle = std::atan2(sl, al);
        float cos = std::cos(halfAngle);
        float sin = std::sin(halfAngle);

        outerVerts.push_back((cos * nU - sin * nV) * offset + bU);
        outerVerts.push_back((sin * nU + cos * nV) * offset + bV);
    }
    return outerVerts;
}


std::vector<float> shellUvGridPos(
    const std::vector<float>& uvs,
    const std::vector<std::vector<int>>& grid,
    const EdgeVector& edges,
    float offset
){
    int numBridgeSegs = grid.size() - 1;
    auto& bVerts = grid[0];
    auto& eVerts = grid[numBridgeSegs];

    std::vector<float> ret(uvs);
    ret.insert(ret.end(), uvs.begin(), uvs.end());
    ret.resize(ret.size() + numBridgeSegs * bVerts.size() * 2);

    std::vector<float> outerVerts = _getOffsettedUvs(uvs, grid, edges, offset);
    for (int i = 0; i < eVerts.size(); ++i){
        ret[eVerts[i] * 2] = outerVerts[i * 2];
        ret[eVerts[i] * 2 + 1] = outerVerts[i * 2 + 1];
    }

    for (int segIdx = 1; segIdx < numBridgeSegs; ++segIdx) {
        float perc = (float)segIdx / (float)(grid.size() - 1);
        for (int vIdx = 0; vIdx < outerVerts.size() / 2; ++vIdx) {
            int tar = grid[segIdx][vIdx] * 2;
            int iSrc = bVerts[vIdx] * 2;
            int oSrc = vIdx * 2;
            for (int i = 0; i < 2; ++i) {
                ret[tar + i] = uvs[iSrc + i] * (1.0 - perc) + outerVerts[oSrc + i] * perc;
            }
        }
    }


    return ret;
}


std::vector<float> shellVertGridPos(
    const std::vector<float>& verts,
    const std::vector<float>& normals,
    const std::vector<std::vector<int>>& grid,
    float innerOffset,
    float outerOffset
){
    int numBridgeSegs = grid.size() - 1;
    auto& bVerts = grid[0];
    auto& eVerts = grid[numBridgeSegs];

    int bVertCount = (numBridgeSegs - 1) * bVerts.size();

    std::vector<float> ret;
    ret.reserve(2 * verts.size() + bVertCount * 3);

    for (int i = 0; i < verts.size(); ++i){
        ret.push_back(verts[i] - normals[i] * outerOffset);
    }
    for (int i = 0; i < verts.size(); ++i){
        ret.push_back(verts[i] + normals[i] * innerOffset);
    }
    ret.resize(ret.size() + bVertCount * 3);

    for (int segIdx = 1; segIdx < numBridgeSegs; ++segIdx){
        float perc = (float)segIdx / (float)(grid.size() - 1);
        for (int vIdx = 0; vIdx < grid[segIdx].size() / 3; ++vIdx){
            int tar = grid[segIdx][vIdx] * 3;
            int iSrc = bVerts[vIdx] * 3;
            int oSrc = eVerts[vIdx] * 3;
            for (int i = 0; i < 3; ++i) {
                ret[tar + i] = verts[iSrc + i] * (1.0 - perc) + verts[oSrc + i] * perc;
            }
        }
    }
    return ret;
}

