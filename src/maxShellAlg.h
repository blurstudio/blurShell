#pragma once

#include <limits>
#include <vector>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include <cmath>


typedef std::pair<size_t, size_t> Edge;
// Use a structure of arrays because I have to 
typedef std::pair<std::vector<size_t>, std::vector<size_t>> EdgeVector;




/*
A hash function for std::pairs taken from boost::hash_combine
*/
struct PairHash {
    template <class T1, class T2>
    size_t operator() (const std::pair<T1, T2> &pair) const {
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
std::vector<size_t> reverseFaces(const std::vector<size_t>& counts, const std::vector<size_t>& faces, size_t offset) {
    std::vector<size_t> ret(faces.size());

    size_t fIdx = 0, rIdx = 0;
    for (auto c: counts){
        ret[rIdx] = faces[fIdx++] + offset;
        for (size_t i = 1; i < c; ++i){
            ret[rIdx + c - i] = faces[fIdx++] + offset;
        }
        rIdx += c;
    }
    return ret;
}




template <typename T>
std::vector<size_t> argsort(const std::vector<T>& v) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    std::sort(idx.begin(), idx.end(),
        [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

    return idx;
}




/*
A quick function to sort edge pairs by their second index
*/
bool eiSort(const std::pair<Edge, size_t>& a, const std::pair<Edge, size_t>& b) {
    return a.first < b.first;
}

/*
Get the index of the next edge around each face
*/
std::vector<size_t> getEdgePairIdxs(const std::vector<size_t>& counts, size_t size) {
    std::vector<size_t> ret(size);
    size_t fIdx = 0, rIdx = 0;
    for (auto c: counts){
        for (size_t i = 1; i < c; ++i){
            ret[rIdx++] = fIdx + i;
        }
        ret[rIdx++] = fIdx;
        fIdx += c;
    }
    return ret;
}


/*
Build a vector of border face-vert edges in the order they appear in the given faces
*/
EdgeVector findFaceVertBorderPairs(const std::vector<size_t>& counts, const std::vector<size_t>& faces) {
    std::vector<size_t> nextIdxs = getEdgePairIdxs(counts, faces.size());
    std::unordered_map<Edge, size_t, PairHash> edgeDict;

    for (size_t eIdx = 0; eIdx < nextIdxs.size(); ++eIdx) {
        auto a = faces[eIdx];
        auto b = faces[nextIdxs[eIdx]];
        Edge key = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
        if (edgeDict.find(key) == edgeDict.end()) {
            edgeDict[key] = eIdx;
        }
        else {
            edgeDict.erase(key);
        }
    }

    // Get the unordered data out of the dict
    std::vector<size_t> startIdxs;
    EdgeVector fvibp;  //face vert index border pairs
    fvibp.first.reserve(edgeDict.size());
    fvibp.second.reserve(edgeDict.size());
    startIdxs.reserve(edgeDict.size());
    for (const auto &it : edgeDict){
        startIdxs.push_back(it.second);
    }

    // Sort the data
    for (auto k : argsort(startIdxs)) {
        fvibp.first.push_back(startIdxs[k]);
        fvibp.second.push_back(nextIdxs[startIdxs[k]]);
    }
    return fvibp;
}

/*
Find the indices where each needle exists in the haystack
If the needle isn't in there, the value will be std::numeric_limits<size_t>::max()
*/
std::vector<size_t> findInArray(const std::vector<size_t>& needles, const std::vector<size_t>& haystack){
    std::unordered_map<size_t, size_t> hayMap;
    for (size_t i = 0; i < haystack.size(); ++i){
        hayMap[haystack[i]] = i;
    }

    std::vector<size_t> ret;
    ret.reserve(needles.size());
    for (auto n : needles) {
        auto hiter = std::find(haystack.begin(), haystack.end(), n);
        size_t val = std::numeric_limits<size_t>::max();
        if (hiter != haystack.end()){
            // Get the index that the iterator is pointing to
            val = hiter - haystack.begin();
        }
        ret.push_back(val);
    }

    return ret;
}


/*
Return the border verts and edge pairs in 3dsMax order
*/
std::tuple<std::vector<size_t>, EdgeVector>
findSortedBorderEdges(
    const std::vector<size_t>& ioCounts,
    const std::vector<size_t>& oFaces,
    const EdgeVector& faceVertIdxBorderPairs
){
    EdgeVector edges;
    edges.first.reserve(faceVertIdxBorderPairs.first.size());
    edges.second.reserve(faceVertIdxBorderPairs.first.size());
    for (size_t i = 0; i < faceVertIdxBorderPairs.first.size(); ++i) {
        auto fi = faceVertIdxBorderPairs.first[i];
        auto si = faceVertIdxBorderPairs.second[i];
        edges.first.push_back(oFaces[fi]);
        edges.second.push_back(oFaces[si]);
    }

    EdgeVector sortedEdges(edges);
    std::sort(sortedEdges.first.begin(), sortedEdges.first.end());
    std::sort(sortedEdges.second.begin(), sortedEdges.second.end());

    std::vector<size_t> stopVals;
    std::set_difference(
        sortedEdges.second.begin(), sortedEdges.second.end(),
        sortedEdges.first.begin(), sortedEdges.first.end(),
        std::back_inserter(stopVals)
    );

    if (stopVals.empty()){
        return {edges.first, edges};
    }

    std::vector<size_t> stopIdxs = findInArray(stopVals, edges.second);

    std::vector<size_t> bVerts;
    bVerts.reserve(edges.first.size() + stopIdxs.size());
    for (size_t p = 0, i = 0; i < edges.first.size(); ++i){
        if (stopIdxs[p] == i){
            bVerts.push_back(edges.second[i]);
            ++ p;
        }
        bVerts.push_back(edges.first[i]);
    }
    return {bVerts, edges};
}


std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<std::vector<size_t>>>
buildBridgesByEdge(
    const std::vector<size_t>& bVerts,
    const EdgeVector& edges,
    size_t bridgeFirstIdx,
    size_t numSegs
){
    EdgeVector edgeInsertIdxs;
    edgeInsertIdxs.second = findInArray(edges.first, bVerts);
    edgeInsertIdxs.first = findInArray(edges.second, bVerts);

    size_t gridHeight = numSegs + 1;
    std::vector<std::vector<size_t>> grid(
        gridHeight,
        std::vector<size_t>(bVerts.size(), std::numeric_limits<size_t>::max())
    );
    grid[0] = bVerts;

    size_t ptr = bridgeFirstIdx;
    for (size_t i = 0; i < edgeInsertIdxs.first.size(); ++i){
        size_t eStart = edgeInsertIdxs.first[i];
        size_t eEnd = edgeInsertIdxs.second[i];
        for (size_t x = 0; x < gridHeight; ++x){
            if (eStart != std::numeric_limits<size_t>::max()) {
                size_t setter = grid[x][eStart];
                if (setter == std::numeric_limits<size_t>::max())
                    grid[x][eStart] = ptr++;
            }
            if (eEnd != std::numeric_limits<size_t>::max()) {
                size_t setter = grid[x][eEnd];
                if (setter == std::numeric_limits<size_t>::max())
                    grid[x][eEnd] = ptr++;
            }
        }
    }

    std::vector<size_t> faces;
    faces.reserve(numSegs * (bVerts.size() - 1));
    for (size_t i = 0; i < grid.size() - 1; ++i){

        /*
        for (size_t j = 0; j < grid[i].size() - 1; ++j){
            auto eis = edgeInsertIdxs.second[j];
            auto eif = edgeInsertIdxs.first[j];

            faces.push_back(grid[i + 1][eis]);
            faces.push_back(grid[i + 1][eif]);
            faces.push_back(grid[i][eif]);
            faces.push_back(grid[i][eis]);
        }
        */

        for (size_t j = 0; j < edgeInsertIdxs.first.size(); ++j){
            auto eis = edgeInsertIdxs.second[j];
            auto eif = edgeInsertIdxs.first[j];

            faces.push_back(grid[i + 1][eis]);
            faces.push_back(grid[i + 1][eif]);
            faces.push_back(grid[i][eif]);
            faces.push_back(grid[i][eis]);
        }



    }

    std::vector<size_t> counts((edges.first.size() * numSegs), 4);
    return {faces, counts, grid};
}


std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<std::vector<size_t>>>
buildBridgesByVert(
    const std::vector<size_t>& bVerts,
    const std::vector<size_t>& eVerts,
    const EdgeVector& edges,
    size_t bridgeFirstIdx,
    size_t numSegs
){
    size_t gridHeight = numSegs + 1;
    std::vector<std::vector<size_t>> grid(gridHeight, std::vector<size_t>(bVerts.size(), std::numeric_limits<size_t>::max()));
    grid[0] = bVerts;
    grid[numSegs] = eVerts;

    for (size_t i = 1; i < numSegs; ++i){
        for (size_t j = 0; j < bVerts.size(); ++j){
            grid[i][j] = j + bridgeFirstIdx + ((i - 1) * bVerts.size());
        }
    }
    EdgeVector edgeInsertIdxs;
    edgeInsertIdxs.second = findInArray(edges.first, bVerts);
    edgeInsertIdxs.first = findInArray(edges.second, bVerts);

    std::vector<size_t> faces;
    faces.reserve(numSegs * (bVerts.size() - 1));
    for (size_t i = 0; i < grid.size() - 1; ++i){
        for (size_t j = 0; j < edgeInsertIdxs.first.size(); ++j){
            auto fj = edgeInsertIdxs.first[j];
            auto sj = edgeInsertIdxs.second[j];

            faces.push_back(grid[i + 1][sj]);
            faces.push_back(grid[i + 1][fj]);
            faces.push_back(grid[i][fj]);
            faces.push_back(grid[i][sj]);
        }
    }

    std::vector<size_t> counts((edges.first.size() * numSegs), 4);
    return {faces, counts, grid};
}



std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<std::vector<size_t>>, EdgeVector, std::vector<size_t>>
shellUvTopo(
    const EdgeVector& faceVertIdxBorderPairs,
    const std::vector<size_t>& oUvFaces,
    const std::vector<size_t>& oUvCounts,
    size_t numUVs,
    size_t numBridgeSegs
){
    auto [bVerts, edges] = findSortedBorderEdges(oUvCounts, oUvFaces, faceVertIdxBorderPairs);

    size_t bridgeFirstIdx = numUVs * 2;
    auto [bFaces, bCounts, grid] = buildBridgesByEdge(
        bVerts, edges, bridgeFirstIdx, numBridgeSegs
    );

    std::vector<size_t> iUvFaces = reverseFaces(oUvCounts, oUvFaces, 0);

    std::vector<size_t> faces(oUvFaces);
    faces.insert(faces.end(), iUvFaces.rbegin(), iUvFaces.rend());
    faces.insert(faces.end(), bFaces.begin(), bFaces.end());

    std::vector<size_t> counts(oUvCounts);
    counts.insert(counts.end(), oUvCounts.begin(), oUvCounts.end());
    counts.insert(counts.end(), bCounts.begin(), bCounts.end());

    return {faces, counts, grid, edges, bVerts};
}


std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<std::vector<size_t>>, std::vector<size_t>>
shellTopo(
    const EdgeVector& faceVertIdxBorderPairs,
    const std::vector<size_t>& oFaces,
    const std::vector<size_t>& oCounts,
    size_t vertCount,
    size_t numBridgeSegs
){
    auto [bVerts, edges] = findSortedBorderEdges(oCounts, oFaces, faceVertIdxBorderPairs);

    std::vector<size_t> eVerts;
    eVerts.reserve(bVerts.size());
    for (size_t i = 0; i < bVerts.size(); ++i){
        eVerts.push_back(bVerts[i] + vertCount);
    }
    size_t bridgeFirstIdx = 2 * vertCount;

    auto [bFaces, bCounts, grid] = buildBridgesByVert(
        bVerts, eVerts, edges, bridgeFirstIdx, numBridgeSegs
    );

    std::vector<size_t> iFaces = reverseFaces(oCounts, oFaces, 0);

    std::vector<size_t> faces(oFaces);
    faces.insert(faces.end(), iFaces.rbegin(), iFaces.rend());
    faces.insert(faces.end(), bFaces.begin(), bFaces.end());
    
    std::vector<size_t> counts(oCounts);
    counts.insert(counts.end(), oCounts.begin(), oCounts.end());
    counts.insert(counts.end(), bCounts.begin(), bCounts.end());

    return {counts, faces, grid, bVerts};
}


std::vector<float> _getOffsettedUvs(
    const std::vector<float>& uvs,
    const std::vector<std::vector<size_t>>& grid,
    const EdgeVector& edges,
    float offset
){
    const auto& bVerts = grid[0];
    



    auto nxtFind = findInArray(bVerts, edges.second);
    std::vector<bool> midPtBool(nxtFind.size(), true);
    std::vector<size_t> nxtIdxs, noNxts;
    nxtIdxs.reserve(nxtFind.size());
    for (size_t i = 0; i < nxtFind.size(); ++i){
        size_t nfi = nxtFind[i];
        if (nfi = std::numeric_limits<size_t>::max()) {
            continue;
        }

        size_t f = edges.first[nfi];
        if (f == std::numeric_limits<size_t>::max()){
            noNxts.push_back(nxtIdxs.size());
            nxtIdxs.push_back(std::numeric_limits<size_t>::max());
            midPtBool[i] = false;
        }
        else {
            nxtIdxs.push_back(edges.first[f]);
        }



    }

    auto prevFind = findInArray(bVerts, edges.first);
    std::vector<size_t> prevIdxs, noPrevs;
    prevIdxs.reserve(prevFind.size());
    for (size_t i = 0; i < prevFind.size(); ++i){
        size_t f = edges.second[prevFind[i]];
        if (f == std::numeric_limits<size_t>::max()){
            noPrevs.push_back(prevIdxs.size());
            prevIdxs.push_back(std::numeric_limits<size_t>::max());
            midPtBool[i] = false;
        }
        else {
            prevIdxs.push_back(edges.second[f]);
        }
    }

    std::vector<size_t> midPts;
    for (size_t i = 0; i < midPtBool.size(); ++i){
        if (midPtBool[i]){
            midPts.push_back(i);
        }
    }

    std::vector<float> outerVerts;
    for (size_t i = 0; i < bVerts.size(); ++i){
        size_t pI, nI, bI;
        float pU, pV, nU, nV, bU, bV;
        pI = prevIdxs[i];
        nI = nxtIdxs[i];
        bI = bVerts[i];

        bU = uvs[bI * 2];
        bV = uvs[bI * 2 + 1];

        if (pI != std::numeric_limits<size_t>::max()){
            pU = uvs[pI * 2] - bU;
            pV = uvs[pI * 2 + 1] - bV;
            float pLen = std::sqrt(pU * pU + pV * pV);
            pU /= pLen;
            pV /= pLen;
            if (nI == std::numeric_limits<size_t>::max()){
                outerVerts.push_back(-pV * offset + bU);
                outerVerts.push_back(pU * offset + bV);
                continue;
            }
        }
        if (nI != std::numeric_limits<size_t>::max()){
            nU = uvs[nI * 2] - bU;
            nV = uvs[nI * 2 + 1] - bV;
            float nLen = std::sqrt(nU * nU + nV * nV);
            nU /= nLen;
            nV /= nLen;
            if (pI == std::numeric_limits<size_t>::max()){
                outerVerts.push_back(nV * offset + bU);
                outerVerts.push_back(-nU * offset + bV);
                continue;
            }
        }

        if ((pI == std::numeric_limits<size_t>::max()) || (nI == std::numeric_limits<size_t>::max())) continue;

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
    const std::vector<std::vector<size_t>>& grid,
    const EdgeVector& edges,
    float offset
){
    size_t numBridgeSegs = grid.size() - 1;
    auto& bVerts = grid[0];
    auto& eVerts = grid[numBridgeSegs];

    std::vector<float> ret(uvs);
    ret.insert(ret.end(), uvs.begin(), uvs.end());
    ret.resize(ret.size() + numBridgeSegs * bVerts.size() * 2);

    std::vector<float> outerVerts = _getOffsettedUvs(uvs, grid, edges, offset);
    for (size_t i = 0; i < eVerts.size(); ++i){
        ret[(size_t)eVerts[i] * 2] = outerVerts[i * 2];
        ret[(size_t)eVerts[i] * 2 + 1] = outerVerts[i * 2 + 1];
    }

    for (size_t segIdx = 1; segIdx < numBridgeSegs; ++segIdx) {
        float perc = (float)segIdx / (float)(grid.size() - 1);
        for (size_t vIdx = 0; vIdx < outerVerts.size() / 2; ++vIdx) {
            size_t tar = (size_t)grid[segIdx][vIdx] * 2;
            size_t iSrc = (size_t)bVerts[vIdx] * 2;
            size_t oSrc = vIdx * 2;
            for (size_t i = 0; i < 2; ++i) {
                ret[tar + i] = (float)uvs[iSrc + i] * (1.0f - perc) + (float)outerVerts[oSrc + i] * perc;
            }
        }
    }


    return ret;
}


std::vector<float> shellVertGridPos(
    const std::vector<float>& verts,
    const std::vector<float>& normals,
    const std::vector<std::vector<size_t>>& grid,
    float innerOffset,
    float outerOffset
){
    size_t numBridgeSegs = grid.size() - 1;
    auto& bVerts = grid[0];
    auto& eVerts = grid[numBridgeSegs];

    size_t bVertCount = (numBridgeSegs - 1) * bVerts.size();

    std::vector<float> ret;
    ret.reserve(2 * verts.size() + bVertCount * 3);

    for (size_t i = 0; i < verts.size(); ++i){
        ret.push_back(verts[i] - normals[i] * outerOffset);
    }
    for (size_t i = 0; i < verts.size(); ++i){
        ret.push_back(verts[i] + normals[i] * innerOffset);
    }
    ret.resize(ret.size() + bVertCount * 3);

    for (size_t segIdx = 1; segIdx < numBridgeSegs; ++segIdx){
        float perc = (float)segIdx / (float)(grid.size() - 1);
        for (size_t vIdx = 0; vIdx < grid[segIdx].size() / 3; ++vIdx){
            size_t tar = (size_t)grid[segIdx][vIdx] * 3;
            size_t iSrc = (size_t)bVerts[vIdx] * 3;
            size_t oSrc = (size_t)eVerts[vIdx] * 3;
            for (size_t i = 0; i < 3; ++i) {
                ret[tar + i] = (float)verts[iSrc + i] * (1.0f - perc) + (float)verts[oSrc + i] * perc;
            }
        }
    }
    return ret;
}

