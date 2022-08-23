import numpy as np


def _lens(vecs):
    """Get the lengths of a bunch of vectors
    Use the einsum instead of the dot product because its
    it's almost always faster than np.dot
    """
    return np.sqrt(np.einsum('...ij,...ij->...i', vecs, vecs))


def _norm(vecs):
    """Normalize an array of vectors"""
    return vecs / _lens(vecs)[..., None]


def buildNormals(anim, counts, faces):
    """Build the per-vertex normals for a mesh

    Arguments:
        anim (np.array): A (frames, vertIdx, component) array of vertex positions
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert

    Returns:
        np.array: A (frames, vertIdx, component) array of per-vertex normals
    """
    # Get all adjacent vertex triples on the mesh
    nextStep = np.arange(len(faces)) + 1

    ranges = np.r_[0, counts.cumsum()]
    ends = ranges[1:] - 1
    nextStep[ends] = ranges[:-1]

    trips = np.empty((len(faces), 3), dtype=int)
    step = faces[nextStep]
    trips[:, 0] = faces
    trips[:, 1] = step
    trips[:, 2] = step[nextStep]

    # Get each edge as animated vectors
    anim = anim.swapaxes(0, 1)
    v1 = anim[trips[:, 0]] - anim[trips[:, 1]]
    v2 = anim[trips[:, 2]] - anim[trips[:, 1]]

    # Possibly get the triangle areas for weighting
    # Technically I should divide by 2, but since the ratio is what matters
    # I don't actually need to do that

    # 3dsMax doesn't take the triangle areas into account when building the
    # normals, but I could if I wanted to
    # triAreas = _lens(np.cross(v1, v2))

    # TODO: Handle zero vectors

    # Normalize the vectors
    v1 = _norm(v1)
    v2 = _norm(v2)

    # expand normalizing the face-vert Normals
    # because we can use the crossProd length to determine
    # the angle between the edges
    fvn = np.cross(v2, v1)
    fvnLens = _lens(fvn)
    fvn = fvn / fvnLens[..., None]
    angles = np.arcsin(fvnLens)

    # Weight by the angles
    weighted = fvn * angles[..., None]

    # Now use a ufunc to sum things based on the center vert
    out = np.zeros(anim.shape)
    np.add.at(out, trips[:, 1], weighted)
    out = _norm(out)
    out[np.isnan(out)] = 0.0
    out = out.swapaxes(0, 1)
    return out


def triangulateFaces(counts, faces):
    """Given the counts and faces of a mesh, return
    a faces array of the triangulated mesh

    Arguments:
        counts (np.array): The number of vertices per face
        faces (np.array): The indices of vertices per face, flattened

    Returns:
        np.array: An N*3 array of vertex indices that will build a fully
            triangulated mesh
        np.array: An N*3 array of indices into the faces array saying which
            face-vert became this triangle
        np.array: An array of face indices that say which polygon each
            triangle was built from
    """
    # Get the biggest polygon size
    maxC = counts.max()

    # Build a single polygon of that size
    # So like, if the polygon was size 5
    # Build the triangle fan: [0,1,2,  0,2,3,  0,3,4]
    remap = np.zeros(maxC + (maxC - 3) * 2, dtype=int)
    rng = np.arange(len(remap) / 3, dtype=int)
    remap[1::3] = rng + 1
    remap[2::3] = rng + 2

    # build an array of the number of vertices that will be
    # in each fan
    recounts = counts + (counts - 3) * 2

    # Build the indices into the original faces array
    # based on the remap and recounts variables

    # get the pairwise start/end of each fan range
    rccs = np.r_[0, recounts.cumsum()]
    # offset each fan by the cumulative number of vertices
    smear = np.arange(rccs[-1]) - rccs[:-1].repeat(recounts)
    # Get the pairwise start/end of each polygon range
    repc = np.r_[0, counts.cumsum()][:-1]
    # Add the repeated polygon offsets to the fan offsets
    # to get my final indexing into the faces array
    fvIdxs = remap[smear] + repc.repeat(recounts)
    fvIdxs = fvIdxs.reshape((-1, 3))
    # Build the final triangle output
    tris = faces[fvIdxs]

    # also get the face that each tri comes from
    faceIdxs = np.repeat(np.arange(tris.size), counts - 2)

    return tris, fvIdxs, faceIdxs


def reverseFaces(counts, faces):
    """Reverse the winding of the given faces

    Arguments:
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert

    Returns:
        np.array: The new reversed flattened indices of each vert
    """
    # Build idAry to be "the number of steps left/right to get the next index"
    # So if everything was -1, that would mean "grab the previous item in the list"
    # And that works for almost everything, except when we want the next face chunk
    # Then we've gotta step from the beginning of one face, to the end of the next,
    # That means we step the sum of the two adjacent faces (minus 1)
    # Then we make sure to start at the final index of the first face
    # Then just apply that rule to jumble the values of "faces" for the output

    # Start with all -1's
    idAry = np.ones(len(faces), dtype=int) * -1
    # Get the indices of the last index of each face
    ends = np.r_[0, counts[:-1].cumsum()]
    # Get the 2-face-step sizes
    idAry[ends[1:]] = counts[1:] + counts[:-1] - 1
    # Start at the last index of the first face
    idAry[ends[0]] = counts[0] - 1
    # Apply the rules for the output
    return faces[idAry.cumsum()]


def _insertOrErase(edgeDict, ab, xy, idx):
    """ If sorted([a, b]) is a key in the dict remove it from the dict
    Otherwise add it with the unsorted value (a, b). This means that
    only edges with an odd number of adjacent faces (ie, borders) will
    be in the map

    Arguments:
        edgeDict (dict): A dict of {(a,b):(a,b)}
        ab (tuple): An ordered pair of vertex indices connected by an edges
        xy (tuple): The indices of a and b in the face array
    """
    a, b, = ab
    key = (a, b) if a < b else (b, a)
    if key in edgeDict:
        del edgeDict[key]
    else:
        edgeDict[key] = ((ab, xy), idx)


def findSortedBorderEdges(counts, faces):
    """ Find all border edges and their face indices
    But make sure they're in the order that 3dsMax would provide them

    Arguments:
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert

    Returns:
        np.array: An N*2 array of properly ordered vertex pairs
        np.array: An N*2 array of indices into the faces array to
            say which face-verts became these indices
        np.array: An array of length N of the face index that
            each vertex pair originated from
    """
    tris, faceVertIdxs, faceIdxs = triangulateFaces(counts, faces)

    edgeDict = {}
    for i in range(0, len(tris), 3):
        triIdx = i // 3
        _insertOrErase(
            edgeDict,
            (tris[i + 1], tris[i + 0]),
            (faceVertIdxs[i + 1], faceVertIdxs[i + 0]),
            faceIdxs[triIdx],
            i + 0,
        )
        _insertOrErase(
            edgeDict,
            (tris[i + 2], tris[i + 1]),
            (faceVertIdxs[i + 2], faceVertIdxs[i + 1]),
            faceIdxs[triIdx],
            i + 1,
        )
        _insertOrErase(
            edgeDict,
            (tris[i + 0], tris[i + 2]),
            (faceVertIdxs[i + 0], faceVertIdxs[i + 2]),
            faceIdxs[triIdx],
            i + 2,
        )

    # Put the values of the map into an array
    # Then sort that array by the edge index
    idxPairs = sorted(edgeDict.values(), key=lambda x: x[-1])
    # Trim the edge index out
    idxPairs = np.array([i[:-2] for i in idxPairs])
    triIdxs = np.array([i[-2] for i in idxPairs])

    borderVertPairs = idxPairs[:, 0]
    faceIndexPairs = idxPairs[:, 1]

    return borderVertPairs, faceIndexPairs, triIdxs



def getEdgePairs(counts, faces):
    """ Build a 2*N array of ordered edge pairs """
    # We're going to build a list of how many steps to take to get
    # the next face-vertex. Most of those values will be 1 (meaning
    # one step to the right), but at the end of each face, we need
    # to step left back to the beginning

    # Start with all 1's
    idAry = np.ones(len(faces), dtype=int)
    # Get the indices of the last index of each face
    ends = np.r_[0, counts[:-1].cumsum()]
    # Get how many steps back it is to the beginning of the face
    idAry[ends] = -counts
    # apply that offset to an arange to get the index of the next face-vert

    # Now add the step value to the index value of each spot in the array
    # to get the real indices of the next value around each face
    nxtIdxs = np.arange(len(faces), dtype=int) + idAry

    # Then apply that offset to the faces
    nxt = faces[nxtIdxs]

    # Finally get all the pairs as rows in an array
    # by stacking the original faces, and nxt
    pairs = np.stack((faces, nxt), axis=1)
    return pairs



def findSortedBorderEdges_NO_TRI(counts, faces):
    """ Find all border edges and their face indices
    But make sure they're in the order that 3dsMax would provide them

    Arguments:
        counts (np.array): The number of verts per face
        faces (np.array): The flattened indices of each vert

    Returns:
        np.array: An N*2 array of properly ordered vertex pairs
        np.array: An N*2 array of indices into the faces array to
            say which face-verts became these indices
        np.array: An array of length N of the face index that
            each vertex pair originated from
    """
    pairs = getEdgePairs(counts, faces)
    # Build an array that's the faceIndex for each edge pair
    faceIdxs = np.repeat(np.arange(len(counts), dtype=int), counts)
    # Build an array that's the index within the face of each edge pair
    offsets = np.repeat(np.r_[0, np.cumsum(counts)][:-1], counts)
    faceEdgeIdxs = np.arange(len(faceIdxs), dtype=int) - offsets

    # Using a dict because it's WAY easier, and I've done most of the numpy
    # transformations already
    edgeDict = {}
    for eIdx, pair in enumerate(pairs):
        a, b, = pair
        key = (a, b) if a < b else (b, a)
        if key in edgeDict:
            del edgeDict[key]
        else:
            edgeDict[key] = eIdx

    edgeIdxs = np.array(sorted(edgeDict.values()))

    borderVertPairs = pairs[edgeIdxs]
    borderFaceIdxs = faceIdxs[edgeIdxs]
    borderFaceEdgeIdxStarts = faceEdgeIdxs[edgeIdxs]

    return borderVertPairs, borderFaceIdxs, borderFaceEdgeIdxStarts




def buildCycles(edges):
    startMap = {edges[i][0]: i for i in range(len(edges))}
    return np.array([startMap[edge[1]] for edge in edges])


def getBridgeIdx(eIdx, segIdx, numBridgeSegs, vertCount, bIdxs):
    """ This is how we get the bridge indices in c++ """
    if segIdx == 0:
        return bIdxs[eIdx]
    if segIdx == numBridgeSegs:
        return bIdxs[eIdx] + vertCount
    return (2 * vertCount) + eIdx + (len(bIdxs) * (segIdx - 1))



def shellUVs(vertFaces, oFaces, oCounts, uvCount, numBridgeSegs):
    """
    vertFaces (np.ndarray): The numpy array of faces for vertex indices
    oFaces (np.ndarray): The numpy array of faces for UV indices

    """
    # each adjacent pair of indices in this array give the slice range
    # into the vertFaces and oFaces arrays for that face index
    faceRanges = np.r_[0, oCounts.cumsum()][:-1]


    origBorderEdges, origBorderEdgeFaceIdxs, origFaceIdxs = findSortedBorderEdges(oCounts, oFaces)


    pass





def shellTopo(oFaces, ioCounts, vertCount, numBridgeSegs):
    origBorderEdges, origBorderEdgeFaceIdxs, origFaceIdxs = findSortedBorderEdges(ioCounts, oFaces)
    cycle = buildCycles(origBorderEdges)

    borderVerts = [e[0] for e in origBorderEdges]

    brEdges = np.full(numBridgeSegs * len(borderVerts), 4, dtype=int)
    counts = np.concatenate((ioCounts, ioCounts, brEdges), axis=0)

    numBorders = len(borderVerts)
    offset = 2 * vertCount

    oge, ogs = np.ogrid[:numBorders, :numBridgeSegs]
    bFaces = np.zeros((numBorders, numBridgeSegs, 4), dtype=int)

    # Build the bridge indices
    bFaces[:, :, 0] = offset + oge + (numBorders * (ogs - 1))
    bFaces[:, :, 1] = offset + cycle[oge] + (numBorders * (ogs - 1))
    bFaces[:, :, 2] = offset + cycle[oge] + (numBorders * ogs)
    bFaces[:, :, 3] = offset + oge + (numBorders * ogs)

    # Connect them back to the original meshes
    bFaces[:, 0, 0] = borderVerts[oge].flatten()
    bFaces[:, 0, 1] = borderVerts[cycle[oge]].flatten()
    bFaces[:, -1, 2] = borderVerts[cycle[oge]].flatten() + vertCount
    bFaces[:, -1, 3] = borderVerts[oge].flatten() + vertCount

    iFaces = reverseFaces(ioCounts, oFaces) + vertCount
    faces = np.concatenate((oFaces, iFaces, bFaces))

    return counts, faces, borderVerts







def shellGeo(rawAnim, normals, bIdxs, numBridgeSegs, innerOffset, outerOffset):
    numFrames = len(rawAnim)

    bVertCount = (numBridgeSegs - 1) * bIdxs.length()
    bVerts = np.zeros((numFrames, bVertCount, 3))
    ret = np.concatenate(
        (
            rawAnim + normals * outerOffset,
            rawAnim - normals * innerOffset,
            bVerts,
        )
    )

    innerVerts = ret[:, bIdxs]
    outerVerts = ret[:, bIdxs + len(rawAnim)]
    for segIdx in range(1, numBridgeSegs):
        perc = float(segIdx) / numBridgeSegs
        ring = innerVerts * (1.0 - perc) + outerVerts * perc
        ringIdxs = bIdxs + ((2 * len(rawAnim)) + (len(bIdxs) * (segIdx - 1)))
        ret[:, ringIdxs] = ring

    return ret


def shell(anim, faces, counts, innerOffset, outerOffset, bridgeSegs):
    vertCount = len(anim[1])
    sCounts, sFaces, borderIdxs = shellTopo(faces, counts, vertCount, bridgeSegs)
    normals = buildNormals(anim, counts, faces)
    sAnim = shellGeo(anim, normals, borderIdxs, bridgeSegs, innerOffset, outerOffset)

    return sAnim, sFaces, sCounts

